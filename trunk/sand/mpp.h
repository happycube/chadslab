typedef unsigned char uint8;
typedef unsigned char byte;
typedef unsigned char bool;
typedef unsigned long uint32;
typedef long int32;

static const bool true = 1;
static const bool false = 0;

enum EError {
	EAGAIN = 0,
	EWOULDBLOCK = -1,
	EEOF = -2,
	ECLOSED = -3,
	ENOANSWER = -42,
	EONFIRE = -51,
};

static inline uint32 r(uint32 ptr)
{
        return  *(volatile uint32 *)ptr;
}

static inline void w(uint32 ptr, uint32 val)
{
	*(volatile uint32 *)ptr = val;
}

void platform_init();
int uart_setbaud(int n, int baud);
int uart_write(int n, byte *b, int len);
int uart_read(int n, byte *b, int len);
