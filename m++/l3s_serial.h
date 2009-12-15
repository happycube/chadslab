/* Registers for the Stellaris(r) UART */

/* Register names and fields are directly from the LM3S811 data sheet */

#define MUS_UARTDR   0x00

#define MUS_UARTFR   0x18 >> 2
#define MUS_UARTFR_TXFE 0x80
#define MUS_UARTFR_RXFF 0x40
#define MUS_UARTFR_TXFF 0x20
#define MUS_UARTFR_RXFE 0x10
#define MUS_UARTFR_BUSY 0x08

#define MUS_UARTIBRD 0x24 >> 2
#define MUS_UARTFBRD 0x28 >> 2

#define MUS_UARTCRH 0x2C >> 2
#define MUS_UARTCRH_FEN  0x10
#define MUS_UARTCRH_WLEN 0x20 // actually 2 bits

#define MUS_UARTCTL 0x30 >> 2

/*
 * XXX:
 * 
 * - Very skeletal.
 * - No interrupt support yet.
 * - Not even tested on real hardware.
 */

class M_UART_Stellaris : public M_UART {
	protected:
		uint32 *regbase32;
		uint32 uart_sysclock;
		bool setEnable(bool en);
	public:
		M_UART_Stellaris(uint32 clock = 2*(10^7), byte *rb = (byte *)0x4000c000);
		virtual uint32 setBaud(uint32 baud = 115200);
		virtual bool rxPoll();
		virtual int32 Read();
		virtual bool Write(byte val);
};
