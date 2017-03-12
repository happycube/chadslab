These files will let you spin up the fans on an nVidia video card in a headless (nominally) non-X configuration.

My use case:  I have a system with a Supermicro X8DTL-IF board (dual Xeon X5650) and a 1070 Founder's Edition card.  The motherboard only has a small heatsink, and left to it's own devices gets extremely hot.  Since the 1070's fan is close enough to the heatsink, it turns out that running it at 60-100% keeps it in the 40-60C range, which is much better and will let me stretch the most life out of the motherboard.

But I often forget to start those, so I figured I'd automate it (and get a little more systemd experience at the same time) - to install this setup:

cp fancontrol /usr/local/sbin
cp xorg.conf /etc/X11
cp \*.service /etc/systemd/system
cd /etc/systemd/system/multi-user.target.wants
ln -sf ../dummyxserver.service .
ln -sf ../nvidiafancontrol.service .

You may have to manually tell systemd to use the new services, but I *think* rebooting will work fine.

