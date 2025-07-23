package gorgonia

import (
	"github.com/pkg/errors"
	"gorgonia.org/gorgonia/internal/encoding"
	"gorgonia.org/tensor"
)

// pad [padTop,padBottom,padLeft,padRight] or [padTop/padBottom,padLeft/padRight]
func Pad(x *Node, pad []int) (*Node, error) {
	group := encoding.NewGroup("Pad")
	xShape := x.Shape()
	var value interface{}
	// check shape
	if xShape.Dims() != 4 {
		return nil, errors.Errorf("Expected input to have a shape with dimension 4")
	}

	h, w := xShape[2], xShape[3]
	padTop := pad[0]
	padLeft := pad[1]
	padBottom := pad[0]
	padRight := pad[1]
	if len(pad) == 4 {
		padTop = pad[0]
		padBottom = pad[1]
		padLeft = pad[2]
		padRight = pad[3]
	}

	mode := "constant"

	if x.Dtype() == tensor.Float64 {
		value = float64(0)
	} else {
		value = float32(0)
	}

	if h+padTop+padBottom < 0 {
		// error
		return nil, errors.New("Impossible height/pad combination")
	}

	if w+padLeft+padRight < 0 {
		// error
		return nil, errors.New("Impossible width/pad combination")
	}

	op := makePadOp(xShape, padTop, padBottom, padLeft, padRight, mode, value)
	retVal, err := ApplyOp(op, x)
	retVal.groups = retVal.groups.Upsert(group)
	return retVal, err
}
