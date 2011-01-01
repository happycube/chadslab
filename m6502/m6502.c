/* 
 * m6502.c - mini 6502 emulator
 *

Copyright (C) 2009-2010 Chad Page. All rights reserved.

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

   1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
   2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE AUTHOR ``AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE AUTHOR OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

 */

/* Logic based off http://www.llx.com/~nparker/a2/opcodes.html */

#include <stdio.h>
#include "m6502.h"

#ifdef DEBUG
char *tr_ins, *tr_add;
#define ITRACE(a) (tr_ins = a)
#define ATRACE(a) (tr_add = a)
#else
#define ITRACE(a)
#define ATRACE(a)
#endif

u8 a = 0, x = 0, y = 0, flags = 0, sp = 0xfd;
u16 pc;

int trace = 1;

#if 0
u8 mem[65536];

inline unsigned char mread(unsigned short addr)
{
	return mem[addr];
}

inline void mwrite(u16 addr, u8 b)
{
	mem[addr] = b;
}
#endif

inline u8 addr_imm()
{
	u8 rv = mread(pc++); 
	if (trace) printf("imm rv %02x pc %04x\n", rv, pc - 1); 
	return rv;
}

inline short addr_zp(u8 n)
{
	return (mread(pc++) + n) & 0xff; 
}

inline unsigned short addr_zpix()
{
	int baseaddr = (mread(pc++) + x) & 0xff; 
	return (mread(baseaddr) + (mread(baseaddr + 1) << 8)); 
}

inline unsigned short addr_zpiy()
{
	int baseaddr = mread(pc++); 
	return (mread(baseaddr) + (mread(baseaddr + 1) << 8) + y); 
}

inline unsigned short read16(u16 addr)
{
	return mread(addr) + (mread(addr + 1) << 8);
}

inline unsigned short read16p()
{
	int t = (mread(pc++));
	return t + (mread(pc++) << 8);
}

inline u16 addr_abs(u8 n)
{
	u16 rv;

	rv = read16p() + n;
	if (trace) printf("abs rv %02x pc %04x\n", rv, pc - 2); 
	return rv;
}

inline u16 addr_absind(u8 n)
{
	u16 addr, rv;

	addr = mread(pc++);
	addr += (mread(pc++) << 8) + n;
	return mread(addr) + (mread(addr) << 8);
}

/* All register-changing insns change f_zero and f_neg */
inline void set(u8 *reg, u8 n)
{
	*reg = n;
	flags &= ~(F_NEG | F_ZERO);
	
	if (n == 0) {
		flags |= F_ZERO; 
	} else if (n & 0x80) {
		flags |= F_NEG; 
	}
}

inline u8 do_adc(u8 reg, int n)
{
	if (flags & F_BCD) {
		u8 p1 = ((flags & F_CARRY) != 0) + (reg & 0x0f) + (n & 0x0f);
		u8 p2 = (p1 >= 10) + (reg >> 4) + (n >> 4);

		flags &= ~(F_CARRY);
		if (p1 >= 10) p1 -= 10;
		if (p2 >= 10) {flags |= F_CARRY; p2 -= 10;}
	
		return (p2 << 4) + p1;
	} else {
		u8 tmp = (u16)reg + (u16)n + ((flags & F_CARRY) != 0);
	
		flags &= ~(F_OVF | F_CARRY);
		if (tmp < reg) {flags |= F_CARRY;} 
		if ((tmp & 0x80) != (reg & 0x80)) flags |= F_OVF;

		return tmp;
	}
}

inline u8 do_sbc(u8 reg, int n)
{
	if (flags & F_BCD) {
		int nd = ((n >> 4) * 10) + (n & 0x0f);
		int val = ((reg >> 4) * 10) + (reg & 0x0f) - nd;

		flags &= ~F_CARRY;
		if (val < 0) {flags |= F_CARRY; val += 100;}

		return ((val / 10) << 4) + (val % 10);
	} else {
		u8 tmp = reg - (n + ((flags & F_CARRY) == 0));
	
		flags |= F_CARRY;
		if (tmp > reg) {flags &= ~F_CARRY;} 
		flags &= ~(F_OVF);
		if ( ((tmp < 0x80) && (reg >= 0x80)) != (n >= 0x80) ) flags |= F_OVF;

		return tmp;
	}
}

inline void do_cmp(u8 reg, int n)
{
	u8 scratch;

	set(&scratch, (reg - n));

	flags &= ~F_CARRY;
	if (reg >= n) flags |= F_CARRY;

	return;
}

void step()
{
	u8 i, aaa, bbb, cc;
	u8 val;
	u16 addr = 0;

	ATRACE("");
	ITRACE("");

	// get next instruction
	i = mread(pc++);

	/* Most of the set breaks down into aaabbbcc */
	aaa = (i & 0xe0) >> 5;
	bbb = (i & 0x1c) >> 2;
	cc = i & 0x03;

	// cc == 11 is undefined, we're gonna NOP these for now
	if (cc == 3) return;

	// all 0x*2 except A2 are *also* undef/nops - 79 insts down!
//	if ((cc == 2) && (i != 0xa2)) return;

	/* most 0x*0 are jumps of some sort */
	if (((i & 0x0f) == 0x00) && (i != 0x80) && (i != 0xa0) && (i != 0xc0) && (i != 0xe0)) {
		unsigned char f;

		/* The jumps aside from these 4 are branches, form xxy10000.  xx = type, y = match */
		switch (i) {
			case 0x20: // JSR abs
			case 0x00: // BRK
				mwrite(sp-- + 0x100, pc & 0xff); 
				mwrite(sp-- + 0x100, pc >> 8); 
				if (i == 0x20) {
					addr = addr_abs(0); ITRACE("JSR");
				} else {
					addr = read16(0xfffe);	
					mwrite(sp-- + 0x100, flags | 0x20);
				}
				pc = addr;	
				goto finish;
			case 0x40: // RTI
				flags = mread(++sp + 0x100);
				ITRACE("RTI");
			case 0x60: // RTS
	                        pc = mread(++sp + 0x100) << 8;
       		                pc += mread(++sp + 0x100);
				goto finish;
			default: break;
		}
		u16 v = mread(pc++);
		switch (i & 0xdf) {
			case 0x10: f = !(flags & F_NEG); break;
			case 0x50: f = !(flags & F_OVF); break;
			case 0x90: f = !(flags & F_CARRY); break;
			case 0xd0: f = !(flags & F_ZERO); break;
		}
		if (i & 0x20) f = !f;
		if (f) {
			if (v & 0x80) 
				pc -= (0x100 - v);
			else
				pc += (v & 0x7f);
		}
		goto finish;
	}
notjump:
	/* single-byte instructions are 0x*8 and 0x[8-e]a */
	if ((i & 0x0f) == 0x08) {	
		switch (i >> 4) {
			case 0x0: mwrite(sp-- + 0x100, flags | 0x20); goto finish;  
			case 0x2: flags = mread(++sp + 0x100); goto finish;  
			case 0x4: mwrite(sp-- + 0x100, a); goto finish;  
			case 0x6: set(&a, mread(++sp + 0x100)); goto finish;  
			case 0x8: set(&y, y - 1); goto finish;
			case 0x9: set(&a, y); goto finish;
			case 0xa: set(&y, a); goto finish;
			case 0xc: set(&y, y + 1); goto finish;
			case 0xe: set(&x, x + 1); goto finish;
			case 0x1: flags &= ~F_CARRY; goto finish; // CLC
			case 0x3: flags |= F_CARRY; goto finish; // SEC
			case 0x5: flags &= ~F_INTDIS; goto finish; // CLI
			case 0x7: flags |= F_INTDIS; goto finish; // SEI
			case 0xB: flags &= ~F_OVF; goto finish; // CLV
			case 0xD: flags &= ~F_BCD; goto finish; // CLD
			case 0xF: flags |= F_BCD; goto finish; // SED
			default: return;
		}
	}

	if ((i & 0x0f) == 0x0a) {
		switch (i >> 4) {
			case 0x8: set(&a, x); goto finish;
			case 0x9: set(&sp, x); goto finish;
			case 0xa: set(&x, a); goto finish;
			case 0xb: set(&x, sp); goto finish;
			case 0xc: set(&x, x - 1); goto finish;
			case 0xd: goto finish;
			case 0xe: goto finish;
			case 0xf: y = a; goto finish; // 65C02
		}
	}

	if (cc == 0x01) {
		switch (bbb) {
			case 0x00: addr = addr_zpix(); ATRACE("zpi, x"); break; 
			case 0x01: addr = addr_zp(0); ATRACE("zp"); break; 
			case 0x02: val = addr_imm(); ATRACE("imm"); break; 
			case 0x03: addr = addr_abs(0); ATRACE("abs"); break; 
			case 0x04: addr = addr_zpiy(); ATRACE("(zp, y)"); break; 
			case 0x05: addr = addr_zp(x); ATRACE("zp, x"); break; 
			case 0x06: addr = addr_abs(y); ATRACE("abs, y"); break; 
			case 0x07: addr = addr_abs(x); ATRACE("abs, x"); break; 
			default: break;
		};
		if (bbb != 0x02) { 
			val = mread(addr);
		} else if (aaa == 0x04) goto finish;

		switch (aaa) {
			case 0x00: set(&a, a | val); break;
			case 0x01: set(&a, a & val); break;
			case 0x02: set(&a, a ^ val); break;
			case 0x03: set(&a, do_adc(a, val)); break;
			case 0x04: mwrite(addr, a); break;
			case 0x05: set(&a, val); break;
			case 0x06: do_cmp(a, val); break; 
			case 0x07: set(&a, do_sbc(a, val)); break; 
			default: break;
		};
		goto finish;
	} 
		
	if ((i & 0xdf) == 0x4c) { // 0x4c:JMP abs 0x6c:JMP (abs)
		addr = addr_abs(0);
		if (i == 0x6c) {
			addr = read16(addr);
		}
		pc = addr;
		return;
	}

	// these addressing modes are used for both cc == 0x02 and cc == 0x03	
	switch (bbb) {
		case 0x01: addr = addr_zp(0); ATRACE("zp"); break; 
		case 0x03: addr = addr_abs(0); ATRACE("abs"); break; 
		case 0x04: addr = addr_zpiy(); ATRACE("(zp, y)"); break; 
		case 0x05: addr = addr_zp(x); ATRACE("zp, x"); break; 
		case 0x07: addr = addr_abs(x); ATRACE("abs, x"); break; 
		default: break;
	};

	if (bbb == 0x00) {
		val = addr_imm(); 
		ATRACE("imm2");
	} else if (bbb == 0x02) {
		val = a; // not used when cc == 0x03	
	} else {
		val = mread(addr);
	}
	
	if (cc == 0x02) {
		u8 out, oc = (flags & F_CARRY) != 0;

		if (aaa < 0x04) {
			u8 cbit, ocbit = 0;

			cbit = (aaa & 0x02) ? 0x01 : 0x80;
			if (aaa & 0x01) 
				ocbit = (aaa & 0x02) ? 0x80 : 0x01;

			flags &= ~F_CARRY;
			if (val & cbit)
				flags |= F_CARRY;

			if (aaa & 0x02)
				val = val >> 1;
			else
				val = val << 1;
		
			if (oc) val |= ocbit;
	
			if (bbb == 0x02) 
				set(&a, val); 
			else {
				set(&out, val); 
				mwrite(addr, out);
			}
	
			goto finish;
		}

		// STX, LDX
		if (aaa < 0x06) {
			// override addressing modes - y-based
			if (bbb == 0x05 || bbb == 0x07) {
				addr = addr - x + y;
				val = mread(addr);
			}

			if (aaa == 0x04) 
				mwrite(addr, x);
			else // 0x05
				set(&x, val);

			goto finish;
		}

		// DEC, INC	
		set(&out, val + ((aaa == 0x06) ? -1 : 1));
		mwrite(addr, out);
		return;
	}

	if (cc == 0x00) {
		if (bbb == 0x02) return;
		if (aaa == 0x01) { // BIT
			flags &= ~(F_ZERO | F_OVF | F_NEG); 
			if (val & 0x80) flags |= F_NEG;
			if (val & 0x40) flags |= F_OVF;
			if (!(a & val)) flags |= F_ZERO;
			goto finish;
		}
		if ((aaa & 0x6) == 0x02) { // JMP
			// handle jmp (abs)
			if (aaa == 3) {
				addr = read16(addr);
			}
			pc = addr;
			goto finish;
		}
		switch (aaa) {
			case 4: mwrite(addr, y); ITRACE("STY"); break;
			case 5: set(&y, val); ITRACE("LDY"); break;
			case 6: do_cmp(y, val); ITRACE("CPY"); break;
			case 7: do_cmp(x, val); ITRACE("CPX"); break;
			default: break;
		}
		goto finish;
	}

finish: 
	; 
#ifdef DEBUG
	printf("%s %s\n", tr_ins, tr_add);
#endif
}
 
