scalar_muladd_fp32:
	push   %rbp
	mov    %rsp,%rbp
	sub    $0x40,%rsp
	movss  0x7aaf(%rip),%xmm0         ; 0x7ff792609050 <ITERATION_COUNT+80>
	movss  %xmm0,-0x14(%rbp)
	movss  0x7aa6(%rip),%xmm0         ; 0x7ff792609054 <ITERATION_COUNT+84>
	movss  %xmm0,-0x18(%rbp)
	movss  0x7a9d(%rip),%xmm0         ; 0x7ff792609058 <ITERATION_COUNT+88>
	movss  %xmm0,-0x1c(%rbp)
	pxor   %xmm0,%xmm0
	movss  %xmm0,-0x4(%rbp)
	movq   $0x0,-0x10(%rbp)
	jmp    0x7ff79260162e             ; <scalar_muladd_fp32+157>
	call   0x7ff792607780             ; <rand>
	movslq %eax,%rdx
	imul   $0x10624dd3,%rdx,%rdx
	shr    $0x20,%rdx
	sar    $0x6,%edx
	mov    %eax,%ecx
	sar    $0x1f,%ecx
	sub    %ecx,%edx
	imul   $0x3e8,%edx,%ecx ; signed multiplication of two operands
	sub    %ecx,%eax
	mov    %eax,%edx
	pxor   %xmm0,%xmm0  ; xmmX - an SSE registers
	cvtsi2ss %edx,%xmm0 ; Convert Doubleword Integer to Scalar Single-Precision Floating-Point Value
	movss  %xmm0,-0x20(%rbp)
	movss  -0x14(%rbp),%xmm0
	mulss  -0x18(%rbp),%xmm0
	addss  -0x1c(%rbp),%xmm0
	addss  -0x20(%rbp),%xmm0
	movss  -0x4(%rbp),%xmm1
	addss  %xmm1,%xmm0  ; Add Scalar Single-Precision Floating-Point Values - SSE
	movss  %xmm0,-0x4(%rbp)
	addq   $0x1,-0x10(%rbp)
	mov    $0x186a0,%eax
	cmp    %rax,-0x10(%rbp)
	jl     0x7ff7926015d3             ; <scalar_muladd_fp32+66>
	movss  -0x4(%rbp),%xmm0
	add    $0x40,%rsp
	pop    %rbp
	ret