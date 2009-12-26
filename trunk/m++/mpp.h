typedef unsigned char uint8;
typedef unsigned char byte;
typedef unsigned long uint32;
typedef long int32;

class MObject {
	public:
		MObject() {};
};

enum EError {
	EAGAIN = 0,
	EWOULDBLOCK = -1,
	EEOF = -2,
	ECLOSED = -3,
	ENOANSWER = -42,
	EONFIRE = -51,
};

class IFileIO {
	public:
		virtual int write(byte *ptr, int num) = 0;
		virtual int read(byte *ptr, int num) = 0;
		virtual int wait() = 0;
};

class IPacketIO {
	public:
		virtual int send(byte *ptr, int len) = 0;
		virtual int recv(byte *ptr, int len) = 0;
};

class MDevice : public MObject {
	protected:
		/* Most devices have a register block */
		byte *regbase;
		int   regsize;
		int   blocking;
	public:
		MDevice() : MObject() {blocking = 0;};
		/* These return 1 if blocking is enabled/possible */
		virtual int setBlock() {return 0;}
		virtual int getBlock() {return blocking;}
};

class M_UART : public MDevice, IFileIO {
	public:
		M_UART() : MDevice() {};
		virtual uint32 setBaud(uint32 baud) = 0;
};

struct EthernetAddress {
	byte addr[6];
};

class M_Ethernet : public MDevice, IPacketIO {
	protected:
	public:
		virtual void setAddress(EthernetAddress &naddr) = 0;
};

inline uint32 r(uint32 ptr)
{
        return  *(volatile uint32 *)ptr;
}

inline void w(uint32 ptr, uint32 val)
{
        *(volatile uint32 *)ptr = val;
}

