package main

import (
	"github.com/ngaut/log"
	G "gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

func main() {
	g := G.NewGraph()
	xData := []float64{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16}
	xVal := tensor.New(tensor.WithShape(1, 1, 4, 4), tensor.WithBacking(xData)) // batch channel height width
	x := G.NewTensor(g, tensor.Float64, 4, G.WithValue(xVal), G.WithName("x"))

	z, err := G.Pad(x, []int{1, 2, 1, 3})
	if err != nil {
		log.Error(err)
	}
	log.Debug(z.Shape())
	log.Debug(z.Value())
	vm := G.NewTapeMachine(g)
	defer vm.Close()
	for i := 0; i < 1; i++ {
		if err := vm.RunAll(); err != nil {
			log.Fatal(err)
		}

		//xData = []float64{1, 2, 3, 4, 5, 6, 7, 8, 9}
		//xVal = tensor.New(tensor.WithShape(3, 3), tensor.WithBacking(xData))
		//G.Let(x, xVal)
		log.Debug(z.Value())
		vm.Reset()
	}
}
