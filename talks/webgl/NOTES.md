
WebGL talk, 2018/01/08 in Santa Barbara

This is a *very* casual/light talk about WebGL and visualization.  
I'm hoping it'll be fun in person, even with the rain...

My original idea was to pull up some local datasets in Jupyter... but unfortunately 
there really aren't any great demos I could find - and I didn't have nearly enough time
to make a new one!

---

This is the late 10's... if you're going to give a talk about something common, 
there's almost certainly a slide deck or few already online you can steal^Wuse. ;)

(Refunds are available at the door.  But the Pizza is non-returnable.)

so we'll see: https://prezi.com/3_kflsippk-x/beneath-the-surface-webgl/

---

shadertoy 

A lot of good shaders here - and you can edit them interactively, great for a demo!

(slow but nice: https://www.shadertoy.com/view/4tByz3)

---

Of course, there are several different frameworks on top of WebGL worth checking out:

- PhiloGL

- deck.gl

- keras.js - training models
  https://transcranial.github.io/keras-js/#/imdb-bidirectional-lstm - one fun demo

---

near-Future:  webgl + webassembly

Far-future: webgl based on Vulkan w/compute - maybe a blend with Apple's WebGPU?

https://architosh.com/2017/09/a-unified-graphics-future-how-the-khronos-group-intends-to-get-us-there-part-2/

Apple being a problem here - does not support OpenGL ES > 3.0, pushing Metal API instead.

(also it turns out for some compute stuff the 3.1 shaders aren't that fast anyway... kludging things with fragment shaders can be faster)


