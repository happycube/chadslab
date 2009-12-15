typedef unsigned char uint8;
typedef unsigned char byte;
typedef unsigned long uint32;
typedef long int32;

class MObject {
	public:
		MObject() {};
};

class MDevice : public MObject {
	protected:
		/* Most devices have a register block */
		byte *regbase;
		int   regsize;
	public:
		MDevice() : MObject() {};
};

class M_UART : public MDevice {
	public:
		M_UART() : MDevice() {};
		virtual uint32 setBaud(uint32 baud) = 0;
		virtual bool rxPoll() = 0;
		virtual int32 Read() = 0;
		virtual bool  Write(byte val) = 0;
};
