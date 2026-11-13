#include "textflag.h"

// func dotBlockNEON(q, rows, dst unsafe.Pointer, dim, n int)
//
// For each of n rows: four 4-wide accumulators take 16 floats per step,
// then fold into one sum. dim must be a positive multiple of 16.
//
// The assembler has no mnemonics for the floating-point vector adds, so
// those five instructions are spelled out as machine words.
TEXT ·dotBlockNEON(SB), NOSPLIT, $0-40
	MOVD q+0(FP), R0
	MOVD rows+8(FP), R1
	MOVD dst+16(FP), R2
	MOVD dim+24(FP), R3
	MOVD n+32(FP), R4
	CBZ  R4, done

row:
	MOVD R0, R5 // query cursor
	MOVD R3, R6 // floats left in this row
	VEOR V0.B16, V0.B16, V0.B16
	VEOR V1.B16, V1.B16, V1.B16
	VEOR V2.B16, V2.B16, V2.B16
	VEOR V3.B16, V3.B16, V3.B16

step:
	VLD1.P 64(R5), [V4.S4, V5.S4, V6.S4, V7.S4]
	VLD1.P 64(R1), [V16.S4, V17.S4, V18.S4, V19.S4]
	VFMLA V4.S4, V16.S4, V0.S4
	VFMLA V5.S4, V17.S4, V1.S4
	VFMLA V6.S4, V18.S4, V2.S4
	VFMLA V7.S4, V19.S4, V3.S4
	SUBS $16, R6
	BNE  step

	WORD $0x4E21D400 // fadd  v0.4s, v0.4s, v1.4s
	WORD $0x4E23D442 // fadd  v2.4s, v2.4s, v3.4s
	WORD $0x4E22D400 // fadd  v0.4s, v0.4s, v2.4s
	WORD $0x6E20D400 // faddp v0.4s, v0.4s, v0.4s
	WORD $0x7E30D800 // faddp s0, v0.2s
	FMOVS F0, (R2)
	ADD  $4, R2
	SUBS $1, R4
	BNE  row

done:
	RET
