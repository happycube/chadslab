#ifndef M6502_H
#define M6502_H

typedef unsigned char u8;
typedef unsigned short u16;

extern u8 a, x, y, flags, sp;
extern u16 pc;

extern int trace;

#define F_NEG 0x80
#define F_OVF 0x40
#define F_BRK 0x10
#define F_BCD 0x08
#define F_INTDIS 0x04
#define F_ZERO 0x02
#define F_CARRY 0x01

unsigned char mread(unsigned short addr);
void mwrite(u16 addr, u8 b);

void step();

#endif
