#include <unistd.h>

int main()
{
	char b[2048];
	int rv = 2048;

	while (rv == 2048) {
		if ((rv = read(0, b, 2048)) != 0) {
			rv = write(1, b, 2048);
		}
	}
	return 0;
}
