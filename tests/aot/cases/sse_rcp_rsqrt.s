# requires: intel
# (the 12-bit tables are Intel's; AMD's differ, so elsewhere only the golden file is compared.)
# RCPPS / RSQRTPS / RCPSS / RSQRTSS (and the VEX forms): the 12-bit approximations must match the hardware bit for bit.
.text
.globl _start
_start:
    mov $0x600000, %rdi
    lea k(%rip), %rsi
    movups 0(%rsi), %xmm0
    rcpps %xmm0, %xmm1
    movups %xmm1, 0(%rdi)
    rsqrtps %xmm0, %xmm2
    movups %xmm2, 16(%rdi)
    movups 16(%rsi), %xmm0
    rcpps %xmm0, %xmm1
    movups %xmm1, 32(%rdi)
    rsqrtps %xmm0, %xmm2
    movups %xmm2, 48(%rdi)
    movups 32(%rsi), %xmm0
    rcpps %xmm0, %xmm1
    movups %xmm1, 64(%rdi)
    rsqrtps %xmm0, %xmm2
    movups %xmm2, 80(%rdi)
    movups 48(%rsi), %xmm0
    rcpps %xmm0, %xmm1
    movups %xmm1, 96(%rdi)
    rsqrtps %xmm0, %xmm2
    movups %xmm2, 112(%rdi)
    movups 64(%rsi), %xmm0
    rcpps %xmm0, %xmm1
    movups %xmm1, 128(%rdi)
    rsqrtps %xmm0, %xmm2
    movups %xmm2, 144(%rdi)
    movups 80(%rsi), %xmm0
    rcpps %xmm0, %xmm1
    movups %xmm1, 160(%rdi)
    rsqrtps %xmm0, %xmm2
    movups %xmm2, 176(%rdi)
    movups 96(%rsi), %xmm0
    rcpps %xmm0, %xmm1
    movups %xmm1, 192(%rdi)
    rsqrtps %xmm0, %xmm2
    movups %xmm2, 208(%rdi)
    movups 112(%rsi), %xmm0
    rcpps %xmm0, %xmm1
    movups %xmm1, 224(%rdi)
    rsqrtps %xmm0, %xmm2
    movups %xmm2, 240(%rdi)
    movups 128(%rsi), %xmm0
    rcpps %xmm0, %xmm1
    movups %xmm1, 256(%rdi)
    rsqrtps %xmm0, %xmm2
    movups %xmm2, 272(%rdi)
    movups 144(%rsi), %xmm0
    rcpps %xmm0, %xmm1
    movups %xmm1, 288(%rdi)
    rsqrtps %xmm0, %xmm2
    movups %xmm2, 304(%rdi)
    movups 160(%rsi), %xmm0
    rcpps %xmm0, %xmm1
    movups %xmm1, 320(%rdi)
    rsqrtps %xmm0, %xmm2
    movups %xmm2, 336(%rdi)
    movups 176(%rsi), %xmm0
    rcpps %xmm0, %xmm1
    movups %xmm1, 352(%rdi)
    rsqrtps %xmm0, %xmm2
    movups %xmm2, 368(%rdi)
    movups 192(%rsi), %xmm0
    rcpps %xmm0, %xmm1
    movups %xmm1, 384(%rdi)
    rsqrtps %xmm0, %xmm2
    movups %xmm2, 400(%rdi)
    movups 208(%rsi), %xmm0
    rcpps %xmm0, %xmm1
    movups %xmm1, 416(%rdi)
    rsqrtps %xmm0, %xmm2
    movups %xmm2, 432(%rdi)
    movups 224(%rsi), %xmm0
    rcpps %xmm0, %xmm1
    movups %xmm1, 448(%rdi)
    rsqrtps %xmm0, %xmm2
    movups %xmm2, 464(%rdi)
    movups 240(%rsi), %xmm0
    rcpps %xmm0, %xmm1
    movups %xmm1, 480(%rdi)
    rsqrtps %xmm0, %xmm2
    movups %xmm2, 496(%rdi)
    movups 256(%rsi), %xmm0
    rcpps %xmm0, %xmm1
    movups %xmm1, 512(%rdi)
    rsqrtps %xmm0, %xmm2
    movups %xmm2, 528(%rdi)
    movups 272(%rsi), %xmm0
    rcpps %xmm0, %xmm1
    movups %xmm1, 544(%rdi)
    rsqrtps %xmm0, %xmm2
    movups %xmm2, 560(%rdi)
    movups 288(%rsi), %xmm0
    rcpps %xmm0, %xmm1
    movups %xmm1, 576(%rdi)
    rsqrtps %xmm0, %xmm2
    movups %xmm2, 592(%rdi)
    movups 304(%rsi), %xmm0
    rcpps %xmm0, %xmm1
    movups %xmm1, 608(%rdi)
    rsqrtps %xmm0, %xmm2
    movups %xmm2, 624(%rdi)
    movups 320(%rsi), %xmm0
    rcpps %xmm0, %xmm1
    movups %xmm1, 640(%rdi)
    rsqrtps %xmm0, %xmm2
    movups %xmm2, 656(%rdi)
    movups 336(%rsi), %xmm0
    rcpps %xmm0, %xmm1
    movups %xmm1, 672(%rdi)
    rsqrtps %xmm0, %xmm2
    movups %xmm2, 688(%rdi)
    movups 352(%rsi), %xmm0
    rcpps %xmm0, %xmm1
    movups %xmm1, 704(%rdi)
    rsqrtps %xmm0, %xmm2
    movups %xmm2, 720(%rdi)
    movups 368(%rsi), %xmm0
    rcpps %xmm0, %xmm1
    movups %xmm1, 736(%rdi)
    rsqrtps %xmm0, %xmm2
    movups %xmm2, 752(%rdi)
    movups 384(%rsi), %xmm0
    rcpps %xmm0, %xmm1
    movups %xmm1, 768(%rdi)
    rsqrtps %xmm0, %xmm2
    movups %xmm2, 784(%rdi)
    movups 400(%rsi), %xmm0
    rcpps %xmm0, %xmm1
    movups %xmm1, 800(%rdi)
    rsqrtps %xmm0, %xmm2
    movups %xmm2, 816(%rdi)
    movups 416(%rsi), %xmm0
    rcpps %xmm0, %xmm1
    movups %xmm1, 832(%rdi)
    rsqrtps %xmm0, %xmm2
    movups %xmm2, 848(%rdi)
    movups 432(%rsi), %xmm0
    rcpps %xmm0, %xmm1
    movups %xmm1, 864(%rdi)
    rsqrtps %xmm0, %xmm2
    movups %xmm2, 880(%rdi)
    movups 448(%rsi), %xmm0
    rcpps %xmm0, %xmm1
    movups %xmm1, 896(%rdi)
    rsqrtps %xmm0, %xmm2
    movups %xmm2, 912(%rdi)
    movups 464(%rsi), %xmm0
    rcpps %xmm0, %xmm1
    movups %xmm1, 928(%rdi)
    rsqrtps %xmm0, %xmm2
    movups %xmm2, 944(%rdi)
    movups 480(%rsi), %xmm0
    rcpps %xmm0, %xmm1
    movups %xmm1, 960(%rdi)
    rsqrtps %xmm0, %xmm2
    movups %xmm2, 976(%rdi)
    movups 496(%rsi), %xmm0
    rcpps %xmm0, %xmm1
    movups %xmm1, 992(%rdi)
    rsqrtps %xmm0, %xmm2
    movups %xmm2, 1008(%rdi)
    movups 0(%rsi), %xmm3
    movss 0(%rsi), %xmm4
    movaps %xmm3, %xmm5
    rcpss %xmm4, %xmm5
    movups %xmm5, 1024(%rdi)
    movaps %xmm3, %xmm6
    rsqrtss %xmm4, %xmm6
    movups %xmm6, 1040(%rdi)
    movups 0(%rsi), %xmm3
    movss 4(%rsi), %xmm4
    movaps %xmm3, %xmm5
    rcpss %xmm4, %xmm5
    movups %xmm5, 1056(%rdi)
    movaps %xmm3, %xmm6
    rsqrtss %xmm4, %xmm6
    movups %xmm6, 1072(%rdi)
    movups 0(%rsi), %xmm3
    movss 8(%rsi), %xmm4
    movaps %xmm3, %xmm5
    rcpss %xmm4, %xmm5
    movups %xmm5, 1088(%rdi)
    movaps %xmm3, %xmm6
    rsqrtss %xmm4, %xmm6
    movups %xmm6, 1104(%rdi)
    movups 0(%rsi), %xmm3
    movss 12(%rsi), %xmm4
    movaps %xmm3, %xmm5
    rcpss %xmm4, %xmm5
    movups %xmm5, 1120(%rdi)
    movaps %xmm3, %xmm6
    rsqrtss %xmm4, %xmm6
    movups %xmm6, 1136(%rdi)
    movups 0(%rsi), %xmm3
    movss 16(%rsi), %xmm4
    movaps %xmm3, %xmm5
    rcpss %xmm4, %xmm5
    movups %xmm5, 1152(%rdi)
    movaps %xmm3, %xmm6
    rsqrtss %xmm4, %xmm6
    movups %xmm6, 1168(%rdi)
    movups 0(%rsi), %xmm3
    movss 20(%rsi), %xmm4
    movaps %xmm3, %xmm5
    rcpss %xmm4, %xmm5
    movups %xmm5, 1184(%rdi)
    movaps %xmm3, %xmm6
    rsqrtss %xmm4, %xmm6
    movups %xmm6, 1200(%rdi)
    movups 0(%rsi), %xmm3
    movss 24(%rsi), %xmm4
    movaps %xmm3, %xmm5
    rcpss %xmm4, %xmm5
    movups %xmm5, 1216(%rdi)
    movaps %xmm3, %xmm6
    rsqrtss %xmm4, %xmm6
    movups %xmm6, 1232(%rdi)
    movups 0(%rsi), %xmm3
    movss 28(%rsi), %xmm4
    movaps %xmm3, %xmm5
    rcpss %xmm4, %xmm5
    movups %xmm5, 1248(%rdi)
    movaps %xmm3, %xmm6
    rsqrtss %xmm4, %xmm6
    movups %xmm6, 1264(%rdi)
    movups 0(%rsi), %xmm3
    movss 32(%rsi), %xmm4
    movaps %xmm3, %xmm5
    rcpss %xmm4, %xmm5
    movups %xmm5, 1280(%rdi)
    movaps %xmm3, %xmm6
    rsqrtss %xmm4, %xmm6
    movups %xmm6, 1296(%rdi)
    movups 0(%rsi), %xmm3
    movss 36(%rsi), %xmm4
    movaps %xmm3, %xmm5
    rcpss %xmm4, %xmm5
    movups %xmm5, 1312(%rdi)
    movaps %xmm3, %xmm6
    rsqrtss %xmm4, %xmm6
    movups %xmm6, 1328(%rdi)
    movups 0(%rsi), %xmm3
    movss 40(%rsi), %xmm4
    movaps %xmm3, %xmm5
    rcpss %xmm4, %xmm5
    movups %xmm5, 1344(%rdi)
    movaps %xmm3, %xmm6
    rsqrtss %xmm4, %xmm6
    movups %xmm6, 1360(%rdi)
    movups 0(%rsi), %xmm3
    movss 44(%rsi), %xmm4
    movaps %xmm3, %xmm5
    rcpss %xmm4, %xmm5
    movups %xmm5, 1376(%rdi)
    movaps %xmm3, %xmm6
    rsqrtss %xmm4, %xmm6
    movups %xmm6, 1392(%rdi)
    movups 0(%rsi), %xmm3
    movss 48(%rsi), %xmm4
    movaps %xmm3, %xmm5
    rcpss %xmm4, %xmm5
    movups %xmm5, 1408(%rdi)
    movaps %xmm3, %xmm6
    rsqrtss %xmm4, %xmm6
    movups %xmm6, 1424(%rdi)
    movups 0(%rsi), %xmm3
    movss 52(%rsi), %xmm4
    movaps %xmm3, %xmm5
    rcpss %xmm4, %xmm5
    movups %xmm5, 1440(%rdi)
    movaps %xmm3, %xmm6
    rsqrtss %xmm4, %xmm6
    movups %xmm6, 1456(%rdi)
    movups 0(%rsi), %xmm3
    movss 56(%rsi), %xmm4
    movaps %xmm3, %xmm5
    rcpss %xmm4, %xmm5
    movups %xmm5, 1472(%rdi)
    movaps %xmm3, %xmm6
    rsqrtss %xmm4, %xmm6
    movups %xmm6, 1488(%rdi)
    movups 0(%rsi), %xmm3
    movss 60(%rsi), %xmm4
    movaps %xmm3, %xmm5
    rcpss %xmm4, %xmm5
    movups %xmm5, 1504(%rdi)
    movaps %xmm3, %xmm6
    rsqrtss %xmm4, %xmm6
    movups %xmm6, 1520(%rdi)
    movups 0(%rsi), %xmm3
    movss 64(%rsi), %xmm4
    movaps %xmm3, %xmm5
    rcpss %xmm4, %xmm5
    movups %xmm5, 1536(%rdi)
    movaps %xmm3, %xmm6
    rsqrtss %xmm4, %xmm6
    movups %xmm6, 1552(%rdi)
    movups 0(%rsi), %xmm3
    movss 68(%rsi), %xmm4
    movaps %xmm3, %xmm5
    rcpss %xmm4, %xmm5
    movups %xmm5, 1568(%rdi)
    movaps %xmm3, %xmm6
    rsqrtss %xmm4, %xmm6
    movups %xmm6, 1584(%rdi)
    movups 0(%rsi), %xmm3
    movss 72(%rsi), %xmm4
    movaps %xmm3, %xmm5
    rcpss %xmm4, %xmm5
    movups %xmm5, 1600(%rdi)
    movaps %xmm3, %xmm6
    rsqrtss %xmm4, %xmm6
    movups %xmm6, 1616(%rdi)
    movups 0(%rsi), %xmm3
    movss 76(%rsi), %xmm4
    movaps %xmm3, %xmm5
    rcpss %xmm4, %xmm5
    movups %xmm5, 1632(%rdi)
    movaps %xmm3, %xmm6
    rsqrtss %xmm4, %xmm6
    movups %xmm6, 1648(%rdi)
    movups 0(%rsi), %xmm3
    movss 80(%rsi), %xmm4
    movaps %xmm3, %xmm5
    rcpss %xmm4, %xmm5
    movups %xmm5, 1664(%rdi)
    movaps %xmm3, %xmm6
    rsqrtss %xmm4, %xmm6
    movups %xmm6, 1680(%rdi)
    movups 0(%rsi), %xmm3
    movss 84(%rsi), %xmm4
    movaps %xmm3, %xmm5
    rcpss %xmm4, %xmm5
    movups %xmm5, 1696(%rdi)
    movaps %xmm3, %xmm6
    rsqrtss %xmm4, %xmm6
    movups %xmm6, 1712(%rdi)
    movups 0(%rsi), %xmm3
    movss 88(%rsi), %xmm4
    movaps %xmm3, %xmm5
    rcpss %xmm4, %xmm5
    movups %xmm5, 1728(%rdi)
    movaps %xmm3, %xmm6
    rsqrtss %xmm4, %xmm6
    movups %xmm6, 1744(%rdi)
    movups 0(%rsi), %xmm3
    movss 92(%rsi), %xmm4
    movaps %xmm3, %xmm5
    rcpss %xmm4, %xmm5
    movups %xmm5, 1760(%rdi)
    movaps %xmm3, %xmm6
    rsqrtss %xmm4, %xmm6
    movups %xmm6, 1776(%rdi)
    movups 0(%rsi), %xmm3
    movss 96(%rsi), %xmm4
    movaps %xmm3, %xmm5
    rcpss %xmm4, %xmm5
    movups %xmm5, 1792(%rdi)
    movaps %xmm3, %xmm6
    rsqrtss %xmm4, %xmm6
    movups %xmm6, 1808(%rdi)
    movups 0(%rsi), %xmm3
    movss 100(%rsi), %xmm4
    movaps %xmm3, %xmm5
    rcpss %xmm4, %xmm5
    movups %xmm5, 1824(%rdi)
    movaps %xmm3, %xmm6
    rsqrtss %xmm4, %xmm6
    movups %xmm6, 1840(%rdi)
    movups 0(%rsi), %xmm3
    movss 104(%rsi), %xmm4
    movaps %xmm3, %xmm5
    rcpss %xmm4, %xmm5
    movups %xmm5, 1856(%rdi)
    movaps %xmm3, %xmm6
    rsqrtss %xmm4, %xmm6
    movups %xmm6, 1872(%rdi)
    movups 0(%rsi), %xmm3
    movss 108(%rsi), %xmm4
    movaps %xmm3, %xmm5
    rcpss %xmm4, %xmm5
    movups %xmm5, 1888(%rdi)
    movaps %xmm3, %xmm6
    rsqrtss %xmm4, %xmm6
    movups %xmm6, 1904(%rdi)
    movups 0(%rsi), %xmm3
    movss 112(%rsi), %xmm4
    movaps %xmm3, %xmm5
    rcpss %xmm4, %xmm5
    movups %xmm5, 1920(%rdi)
    movaps %xmm3, %xmm6
    rsqrtss %xmm4, %xmm6
    movups %xmm6, 1936(%rdi)
    movups 0(%rsi), %xmm3
    movss 116(%rsi), %xmm4
    movaps %xmm3, %xmm5
    rcpss %xmm4, %xmm5
    movups %xmm5, 1952(%rdi)
    movaps %xmm3, %xmm6
    rsqrtss %xmm4, %xmm6
    movups %xmm6, 1968(%rdi)
    movups 0(%rsi), %xmm3
    movss 120(%rsi), %xmm4
    movaps %xmm3, %xmm5
    rcpss %xmm4, %xmm5
    movups %xmm5, 1984(%rdi)
    movaps %xmm3, %xmm6
    rsqrtss %xmm4, %xmm6
    movups %xmm6, 2000(%rdi)
    movups 0(%rsi), %xmm3
    movss 124(%rsi), %xmm4
    movaps %xmm3, %xmm5
    rcpss %xmm4, %xmm5
    movups %xmm5, 2016(%rdi)
    movaps %xmm3, %xmm6
    rsqrtss %xmm4, %xmm6
    movups %xmm6, 2032(%rdi)
    vmovups 0(%rsi), %ymm0
    vrcpps %ymm0, %ymm1
    vmovups %ymm1, 0xf00(%rdi)
    vrsqrtps %ymm0, %ymm2
    vmovups %ymm2, 0xf20(%rdi)
    vmovups 32(%rsi), %xmm3
    vrcpps %xmm3, %xmm4
    vmovups %xmm4, 0xf40(%rdi)
    vrcpss 8(%rsi), %xmm3, %xmm5
    vmovups %xmm5, 0xf50(%rdi)
    vrsqrtss 12(%rsi), %xmm3, %xmm6
    vmovups %xmm6, 0xf60(%rdi)
    ret
.balign 16
k:
    .long 0x3f800000
    .long 0x40000000
    .long 0x40400000
    .long 0x3f000000
    .long 0x3fffffff
    .long 0x7f7fffff
    .long 0x00800000
    .long 0x7e800000
    .long 0x7f000000
    .long 0x01000000
    .long 0x7effffff
    .long 0xbf800000
    .long 0xc0000000
    .long 0x80800000
    .long 0x00000000
    .long 0x80000000
    .long 0x7f800000
    .long 0xff800000
    .long 0x7fc00001
    .long 0x7f800001
    .long 0xffc00000
    .long 0xff800001
    .long 0x00000001
    .long 0x007fffff
    .long 0x80000001
    .long 0x807fffff
    .long 0x3f800001
    .long 0x3f801000
    .long 0x7e7fffff
    .long 0x7f7ffffe
    .long 0x00ffffff
    .long 0x01ffffff
    .long 0x29f953a6
    .long 0x269e0d37
    .long 0x335351d2
    .long 0x0c5c7fd0
    .long 0x09e91f84
    .long 0x892f902b
    .long 0x0caecee4
    .long 0x9531985d
    .long 0x07f4712e
    .long 0x81e74ef5
    .long 0x1b84cca8
    .long 0x1600a35a
    .long 0x383586aa
    .long 0x11e20b8f
    .long 0x1f0b9c7b
    .long 0x8d116ece
    .long 0x368790ee
    .long 0xd3ac94af
    .long 0x488fd8be
    .long 0xf28c105d
    .long 0x1d50b859
    .long 0xa09f76b5
    .long 0x4b794e86
    .long 0x0fd630f1
    .long 0x4a4af305
    .long 0x658cda14
    .long 0x06fcf5ed
    .long 0x3898d190
    .long 0x064740cb
    .long 0xdbc496cb
    .long 0x11a511ea
    .long 0x6b4cb242
    .long 0x12c53531
    .long 0x1e27a1c0
    .long 0x49a77c55
    .long 0x8f6d0558
    .long 0x68d74bdd
    .long 0x2e44158b
    .long 0x0dca71df
    .long 0x923a7369
    .long 0x52180c28
    .long 0x5f557203
    .long 0x0cc61c7d
    .long 0xb64ce422
    .long 0x08c83d38
    .long 0x0f4205b4
    .long 0x4f9a5cda
    .long 0x7f150524
    .long 0x57c40f68
    .long 0x6d76b07e
    .long 0x63a835f9
    .long 0x7731af10
    .long 0x4b763353
    .long 0x7403e430
    .long 0x2ea65ec3
    .long 0x3f98e277
    .long 0x66170298
    .long 0xb2f14c94
    .long 0x641f3e8d
    .long 0x14f4733f
    .long 0x4a266e90
    .long 0x86734721
    .long 0x3ff00481
    .long 0x57ee05cd
    .long 0x5db97366
    .long 0x49b64a08
    .long 0x4e7d765e
    .long 0x12bd4ace
    .long 0x0fc18703
    .long 0x6b0a18e8
    .long 0x15e0e9fe
    .long 0x5790f82e
    .long 0x13f77565
    .long 0x7d2caf82
    .long 0x360504be
    .long 0xf646e1f4
    .long 0x5609ef77
    .long 0xc3baea9e
    .long 0x47c958e9
    .long 0xca02135e
    .long 0x70e8bfcd
    .long 0x5051c1cc
    .long 0x2c58ff70
    .long 0x59a54a7b
    .long 0x4cbf930a
    .long 0x9474031b
    .long 0x66ba64ef
    .long 0x119a72d1
    .long 0x6c0bfaf4
    .long 0xf1d69ed6
    .long 0x233caf41
    .long 0xb2715945
    .long 0x558851eb
    .long 0x0f88080b
    .long 0x5e59ca7d
    .long 0x4f426dcb
