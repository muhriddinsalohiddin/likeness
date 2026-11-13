#include "textflag.h"

// func dotBlockAVX2(q, rows, dst unsafe.Pointer, dim, n int)
//
// For each of n rows: four 8-wide accumulators take 32 floats per step,
// then fold into one sum. dim must be a positive multiple of 32.
TEXT ·dotBlockAVX2(SB), NOSPLIT, $0-40
	MOVQ q+0(FP), SI
	MOVQ rows+8(FP), DI
	MOVQ dst+16(FP), DX
	MOVQ dim+24(FP), R8
	MOVQ n+32(FP), CX
	TESTQ CX, CX
	JZ   done

row:
	MOVQ SI, R9  // query cursor
	MOVQ R8, R10 // floats left in this row
	VXORPS Y0, Y0, Y0
	VXORPS Y1, Y1, Y1
	VXORPS Y2, Y2, Y2
	VXORPS Y3, Y3, Y3

step:
	VMOVUPS (R9), Y4
	VMOVUPS 32(R9), Y5
	VMOVUPS 64(R9), Y6
	VMOVUPS 96(R9), Y7
	VFMADD231PS (DI), Y4, Y0
	VFMADD231PS 32(DI), Y5, Y1
	VFMADD231PS 64(DI), Y6, Y2
	VFMADD231PS 96(DI), Y7, Y3
	ADDQ $128, R9
	ADDQ $128, DI
	SUBQ $32, R10
	JNZ  step

	VADDPS Y1, Y0, Y0
	VADDPS Y3, Y2, Y2
	VADDPS Y2, Y0, Y0
	VEXTRACTF128 $1, Y0, X1
	VADDPS X1, X0, X0
	VHADDPS X0, X0, X0
	VHADDPS X0, X0, X0
	VMOVSS X0, (DX)
	ADDQ $4, DX
	DECQ CX
	JNZ  row

done:
	VZEROUPPER
	RET
