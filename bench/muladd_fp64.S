scalar_muladd_fp64:
	push   %rbp
	mov    %rsp,%rbp
	sub    $0x50,%rsp
	movsd  0x7a0c(%rip),%xmm0         ; 0x7ff792609060 <ITERATION_COUNT+96>
	movsd  %xmm0,-0x18(%rbp)
	movsd  0x7a07(%rip),%xmm0         ; 0x7ff792609068 <ITERATION_COUNT+104>
	movsd  %xmm0,-0x20(%rbp)
	movsd  0x7a02(%rip),%xmm0         ; 0x7ff792609070 <ITERATION_COUNT+112>
	movsd  %xmm0,-0x28(%rbp)
	pxor   %xmm0,%xmm0
	movsd  %xmm0,-0x8(%rbp)
	movq   $0x0,-0x10(%rbp)
	jmp    0x7ff7926016e1             ; <scalar_muladd_fp64+157>
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
	cvtsi2sd %edx,%xmm0 ; Convert Doubleword Integer to Scalar Double-Precision Floating-Point Value
	movsd  %xmm0,-0x30(%rbp)
	movsd  -0x18(%rbp),%xmm0
	mulsd  -0x20(%rbp),%xmm0
	addsd  -0x28(%rbp),%xmm0
	addsd  -0x30(%rbp),%xmm0
	movsd  -0x8(%rbp),%xmm1
	addsd  %xmm1,%xmm0  ; Add Scalar Double-Precision Floating-Point Values
	movsd  %xmm0,-0x8(%rbp)
	addq   $0x1,-0x10(%rbp)
	mov    $0x186a0,%eax
	cmp    %rax,-0x10(%rbp)
	jl     0x7ff792601686             ; <scalar_muladd_fp64+66>
	movsd  -0x8(%rbp),%xmm0
	movq   %xmm0,%rax
	movq   %rax,%xmm0
	add    $0x50,%rsp
	pop    %rbp
	ret