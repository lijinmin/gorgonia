package gorgonia

import (
	"fmt"
	"github.com/chewxy/hm"
	"github.com/pkg/errors"
	"gorgonia.org/tensor"
	"hash"
)

type padOp struct {
	padTop, padBottom, padLeft, padRight int
	mode                                 string
	value                                interface{}
	mask                                 tensor.Tensor
}

func makePadOp(inputShape tensor.Shape, padTop, padBottom, padLeft, padRight int, mode string, value interface{}) *padOp {
	op := &padOp{
		padTop:    padTop,
		padBottom: padBottom,
		padLeft:   padLeft,
		padRight:  padRight,
		mode:      mode,
		value:     value,
	}
	op.mask = tensor.New(tensor.Of(tensor.Int), tensor.WithShape(op.calcShape(inputShape)...))
	return op
}

func (op *padOp) Arity() int { return 1 }

func (op *padOp) Type() hm.Type {
	t := makeTensorType(4, hm.TypeVariable('a'))
	return hm.NewFnType(t, t)
}

func (op *padOp) InferShape(inputs ...DimSizer) (tensor.Shape, error) {
	if s, ok := inputs[0].(tensor.Shape); ok {
		return op.calcShape(s), nil
	}
	return nil, errors.Errorf("Expected a shape")
}

func (op *padOp) Do(inputs ...Value) (retVal Value, err error) {
	var in, out tensor.Tensor
	if in, err = op.checkInput(inputs...); err != nil {
		return nil, err
	}
	inShp := in.Shape()
	out = tensor.New(tensor.Of(in.Dtype()), tensor.WithShape(op.calcShape(inShp)...), tensor.WithEngine(in.Engine()))
	op.do(out, in)
	return out, nil
}

func (op *padOp) ReturnsPtr() bool     { return false }
func (op *padOp) CallsExtern() bool    { return false }
func (op *padOp) OverwritesInput() int { return -1 }
func (op *padOp) WriteHash(h hash.Hash) {
	fmt.Fprintf(h, "Pad{%d, %d, %d, %d}( mode: (%s),value: (%f)",
		op.padTop, op.padBottom, op.padLeft, op.padRight,
		op.mode, op.value)
}

func (op *padOp) Hashcode() uint32 { return simpleHash(op) }

func (op *padOp) String() string {
	return fmt.Sprintf("Pad{%d, %d, %d, %d}( mode: (%s),value: (%f)",
		op.padTop, op.padBottom, op.padLeft, op.padRight,
		op.mode, op.value)
}

func (op *padOp) UsePreallocDo(prealloc Value, inputs ...Value) (Value, error) {
	var in tensor.Tensor
	var err error
	if in, err = op.checkInput(inputs...); err != nil {
		return nil, err
	}

	if p, ok := prealloc.(tensor.Tensor); ok {
		op.do(p, in)
		return p, nil
	}
	return nil, errors.Errorf("Expected prealloc to be a tensor")
}

func (op *padOp) checkInput(inputs ...Value) (tensor.Tensor, error) {
	if err := checkArity(op, len(inputs)); err != nil {
		return nil, err
	}

	var in tensor.Tensor
	var ok bool
	if in, ok = inputs[0].(tensor.Tensor); !ok {
		return nil, errors.Errorf("Expected input to be a tensor")
	}

	if in.Shape().Dims() != 4 {
		return nil, errors.Errorf("Expected input to have 4 dimensions")
	}
	return in, nil
}

// calcShape calculates the output shape given an input shape
func (op *padOp) calcShape(s tensor.Shape) tensor.Shape {
	b, c, h, w := s[0], s[1], s[2], s[3]

	pooledH := h + op.padTop + op.padBottom
	pooledW := w + op.padRight + op.padLeft
	return tensor.Shape{b, c, pooledH, pooledW}
}

// do prepares the data, and then dispatches it to the correct (computation) kernel.
// out is the preallocated tensor
func (op *padOp) do(out, in tensor.Tensor) {
	outShape := out.Shape()
	outStride := out.Strides()[1]
	inShape := in.Shape()
	inStride := in.Strides()[1]
	maskStride := op.mask.Strides()[1]

	b, c, h, w := outShape[0], outShape[1], outShape[2], outShape[3]
	inH, inW := inShape[2], inShape[3]

	if op.mask == nil {
		op.mask = tensor.New(tensor.Of(tensor.Int), tensor.WithShape(op.calcShape(inShape)...))
	}

	maskData := op.mask.Data().([]int)

	switch in.Dtype() {
	case tensor.Float64:
		op.f64s(b, c, h, w, inH, inW,
			outStride, inStride, maskStride,
			out.Data().([]float64), in.Data().([]float64),
			maskData)
	case tensor.Float32:
		op.f32s(b, c, h, w, inH, inW,
			outStride, inStride, maskStride,
			out.Data().([]float32), in.Data().([]float32),
			maskData)
	}
}

func (op *padOp) f32s(batches, channels, outH, outW, inH, inW,
	outStride, inStride, maskStride int,
	outData, inData []float32,
	maskData []int) {

	// set values
	for i := range outData {
		outData[i] = op.value.(float32)
		maskData[i] = -1
	}
	startTop := -op.padTop
	endBottom := inH + op.padBottom
	startLeft := -op.padLeft
	endRight := inW + op.padRight

	for b := 0; b < batches; b++ {
		for c := 0; c < channels; c++ {
			outy := 0
			for ph := startTop; ph < endBottom; ph++ {
				outy += 1
				if ph < 0 || ph >= inH {
					continue
				}
				outx := 0
				for pw := startLeft; pw < endRight; pw++ {
					outx += 1
					if pw < 0 || pw >= inW {
						continue
					}
					outIndex := (outy-1)*outW + outx - 1

					inIndex := ph*inW + pw

					outData[outIndex] = inData[inIndex]
				}
			}
			// skip by strides
			inData = inData[inStride:]
			outData = outData[outStride:]
			maskData = maskData[maskStride:]
		}
	}
}

func (op *padOp) f64s(batches, channels, outH, outW, inH, inW,
	outStride, inStride, maskStride int,
	outData, inData []float64,
	maskData []int) {

	// set values
	for i := range outData {
		outData[i] = op.value.(float64)
		maskData[i] = -1
	}
	startTop := -op.padTop
	endBottom := inH + op.padBottom
	startLeft := -op.padLeft
	endRight := inW + op.padRight

	for b := 0; b < batches; b++ {
		for c := 0; c < channels; c++ {
			outy := 0
			for ph := startTop; ph < endBottom; ph++ {
				outy += 1
				if ph < 0 || ph >= inH {
					continue
				}
				outx := 0
				for pw := startLeft; pw < endRight; pw++ {
					outx += 1
					if pw < 0 || pw >= inW {
						continue
					}
					outIndex := (outy-1)*outW + outx - 1

					inIndex := ph*inW + pw

					outData[outIndex] = inData[inIndex]
				}
			}
			// skip by strides
			inData = inData[inStride:]
			outData = outData[outStride:]
			maskData = maskData[maskStride:]
		}
	}
}
