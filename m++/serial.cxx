#include "mpp.h"
#include "l3s_serial.h"

bool M_UART_Stellaris::setEnable(bool en) {
//	regbase32[MUS_UARTCTL] = (regbase32[MUS_UARTCTL] & ~0x01) | (en ? 0x01 : 0x00);
	w(0x4000c030, 0x300 | (en != 0));
	w(0x4000c02c, 0x70);
	if (!en) {
		/* XXX : wait for everything to stop */
//		regbase32[MUS_UARTCRH] &= ~MUS_UARTCRH_FEN;
	}	
	return true;
};

M_UART_Stellaris::M_UART_Stellaris(uint32 clock, byte *rb) : M_UART() {
	uart_sysclock = clock;
	regbase = rb; 

	regbase32 = (uint32 *)regbase; 
	regbase32[MUS_UARTCRH] = 0x60; // N,8,1
	setBaud(115200);
};

uint32 M_UART_Stellaris::setBaud(uint32 baud) {
	uint32 baud16, frac;

	setEnable(false);
	w(0x40004420, 0x03);
	w(0x4000451c, 0x03);
//	w(0x4000c02c, 0x70);
	/* 16mhz values */
//	w(0x4000c024, 0x08);
//	w(0x4000c028, 0x2c); 
	/* 50mhz values */
	w(0x4000c024, 27);
	w(0x4000c028, 8); 
	/* 100mhz values */
//	w(0x4000c024, 54);
//	w(0x4000c028, 16); 
//	w(0x4000c038, 0x50); 

#if 0
	/* This is based on section 12.3 of the LM3S811 DS,
	 * converted into 32-bit integer math */	
	baud16 = (16 * baud);

	regbase32[MUS_UARTIBRD] = (uart_sysclock / baud16);	
	frac = (((uart_sysclock % baud16) * 64) + (baud16 / 2)) / baud16; 
	regbase32[MUS_UARTFBRD] = (frac > 63) ? 63 : frac;
#endif
	setEnable(true);
	return baud;
}

bool M_UART_Stellaris::rxPoll() {
	return !(r(0x4000c018) & MUS_UARTFR_RXFE);
}

int M_UART_Stellaris::read(byte *ptr, int num) {
	/* TODO:  Error handling */
	if (!M_UART_Stellaris::rxPoll()) return 0;
	
	*ptr = r(0x4000c000) & 0xff;
	return 1;
}

int M_UART_Stellaris::write(byte *ptr, int num) {
//	if (regbase32[MUS_UARTFR] && MUS_UARTFR_TXFF)
//		return false;

	w(0x4000c000, (uint32)*ptr);
	return 1;
}

