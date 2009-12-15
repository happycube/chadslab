#include "mpp.h"
#include "l3s_serial.h"

bool M_UART_Stellaris::setEnable(bool en) {
	regbase32[MUS_UARTCTL] = (regbase32[MUS_UARTCTL] & ~0x01) | (en ? 0x01 : 0x00);
	if (!en) {
		/* XXX : wait for everything to stop */
		regbase32[MUS_UARTCRH] &= ~MUS_UARTCRH_FEN;
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
	
	/* This is based on section 12.3 of the LM3S811 DS,
	 * converted into 32-bit integer math */	
	baud16 = (16 * baud);

	regbase32[MUS_UARTIBRD] = (uart_sysclock / baud16);	
	frac = (((uart_sysclock % baud16) * 64) + (baud16 / 2)) / baud16; 
	regbase32[MUS_UARTFBRD] = (frac > 63) ? 63 : frac;
	setEnable(true);
	return baud;
}

bool M_UART_Stellaris::rxPoll() {
	return regbase32[MUS_UARTFR] && MUS_UARTFR_RXFE;	
}

int32 M_UART_Stellaris::Read() {
	/* TODO:  Error handling */
	return (M_UART_Stellaris::rxPoll() ? (regbase32[MUS_UARTDR] & 0xff) : -1);
}

bool M_UART_Stellaris::Write(byte val) {
//	if (regbase32[MUS_UARTFR] && MUS_UARTFR_TXFF)
//		return false;

	regbase32[MUS_UARTDR] = (uint32)val;
	return true;
}

