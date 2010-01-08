#include "mpp.h"
#include "lm3.h"

volatile uint32 x, y;
void platform_init()
{
	/* need settling time after reset */
	for (x = 0; x < 500000; x++);
	w(LM3_SYS_RCC1, LM3_9B92_STARTOSC1);
	/* XXX: need osc to settle in too */
	for (x = 0; x < 500000; x++);
	w(LM3_SYS_RCC2, LM3_9B92_SETOSC2);

	/* power up pll */
	w(LM3_SYS_RCC1, LM3_9B92_STARTPLL1);
	w(LM3_SYS_RCC2, LM3_9B92_STARTPLL2);

	while (!(r(LM3_SYS_RIS) & LM3_SYS_RIS_PLLLOCK));
	/* pll is ready, go to 50mhz */
	w(LM3_SYS_RCC1, LM3_9B92_SETPLL1);
	/* 100mhz oc */
//	w(LM3_SYS_RCC1, 0x00841540);

	w(LM3_SYS_RCGC1, 0x00000001);
	w(LM3_SYS_RCGC2, 0x00000001);
}

bool pl011_setEnable(bool en) {
//	regbase32[MUS_UARTCTL] = (regbase32[MUS_UARTCTL] & ~0x01) | (en ? 0x01 : 0x00);
	w(0x4000c030, 0x300 | (en != 0));
	w(0x4000c02c, 0x70);
	if (!en) {
		/* XXX : wait for everything to stop */
//		regbase32[MUS_UARTCRH] &= ~MUS_UARTCRH_FEN;
	}	
	return true;
};

void uart_setbaud(int n, int baud) {
	uint32 baud16, frac;

	pl011_setEnable(false);
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
	pl011_setEnable(true);
	return 0;
}

int uart_rxpoll(int n) {
	return !(r(0x4000c018) & PL011_FR_RXFE);
}

int uart_read(int n, byte *ptr, int num) {
	/* TODO:  Error handling */
	if (uart_rxpoll(n)) return 0;
	
	*ptr = r(0x4000c000) & 0xff;
	return 1;
}

int uart_write(int n, byte *ptr, int num) {
	int i;
	
	for (i = 0; i < num; i++, ptr++) {
		while (r(0x4000c018) & PL011_FR_TXFF);
		w(0x4000c000, (uint32)*ptr);
	}
	return i;
}

