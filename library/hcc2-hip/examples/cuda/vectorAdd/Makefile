#-----------------------------------------------------------------------
#
#  Makefile: Cuda clang demo Makefile for both amdgcn and nvptx targets.
#            amdgcn targets begin with gfx. nvptx targets begin with sm_
#
#  Run "make help" to see how to use this Makefile
#
#-----------------------------------------------------------------------
# MIT License
# Copyright (c) 2017 Advanced Micro Devices, Inc. All Rights Reserved.
#
# Permission is hereby granted, free of charge, to any person
# obtaining a copy of this software and associated documentation
# files (the "Software"), to deal in the Software without
# restriction, including without limitation the rights to use, copy,
# modify, merge, publish, distribute, sublicense, and/or sell copies
# of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS
# BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN
# ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

TESTNAME =vectorAdd
FILETYPE =cu

ifeq ($(HCC2),)
HCC2 = /opt/rocm/hcc2
endif
ifeq ($(CUDA),)
CUDA = /usr/local/cuda
endif
ifeq ($(TARGETS),)
TARGETS =--cuda-gpu-arch=sm_35
endif

CC       =$(HCC2)/bin/clang++
CFLAGS   =-O3 -std=c++11 -x cuda

ifeq (sm_,$(findstring sm_,$(TARGETS)))
  LFLAGS +=-L$(CUDA)/targets/x86_64-linux/lib -lcudart
  LFLAGS +=-Wl,-rpath,$(CUDA)/targets/x86_64-linux/lib
endif

# You do not need the cuda sdk for gfx targets
ifeq (gfx,$(findstring gfx,$(TARGETS)))
  LFLAGS +=-L/opt/rocm/lib -L$(HCC2)/lib -lhip_hcc -lhc_am -lhiprt
  LFLAGS +=-Wl,-rpath,/opt/rocm/lib -Wl,-rpath,$(HCC2)/lib
endif

# ----- Demo compile and link in one step, no object code saved
$(TESTNAME): $(TESTNAME).$(FILETYPE)
	$(CC) $(CFLAGS) $(TARGETS) $(LFLAGS) $^ -o $@

run: $(TESTNAME)
	./$(TESTNAME)

#  ----   Demo compile and link in two steps, object saved
$(TESTNAME).o: $(TESTNAME).$(FILETYPE)
	$(CC) -c $(CFLAGS) $(TARGETS) $^

obin:	$(TESTNAME).o
	$(CC) $(LFLAGS) $^ -o obin

run_obin: obin
	./obin

#  ----   Demo compile to intermediates LLVMIR or assembly
$(TESTNAME).ll: $(TESTNAME).$(FILETYPE)
	$(CC) -c -S -emit-llvm $(CFLAGS) $(TARGETS) $^

$(TESTNAME).s: $(TESTNAME).$(FILETYPE)
	$(CC)  -c -S $(CFLAGS) $(TARGETS)  $^

help:
	@echo
	@echo "Makefile Help:"
	@echo "  Source:		$(TESTNAME).$(FILETYPE)"
	@echo "  Compiler: 		$(CC)"
	@echo "  Compiler flags: 	$(CFLAGS)"
	@echo
	@echo "Avalable Targets:"
	@echo "  make			// build binary $(TESTNAME)"
	@echo "  make run		// run $(TESTNAME)"
	@echo "  make $(TESTNAME).o	// compile, be, & assemble : -c"
	@echo "  make obin         	// link step only"
	@echo "  make run_obin     	// run obin "
	@echo "  make $(TESTNAME).s	// compile & backend steps : -c -S"
	@echo "  make $(TESTNAME).ll	// compile step only       : -c -S -emit-llvm"
	@echo "  make clean		// cleanup files"
	@echo "  make help 		// this help"
	@echo
	@echo "Environment Variables:"
	@echo "  HCC2      default: /opt/rocm/hcc2        value: $(HCC2)"
	@echo "  CUDA      default: /usr/local/cuda       value: $(CUDA)"
	@echo "  TARGETS   default: --cuda-gpu-arch=gfx803"
	@echo "              value: $(TARGETS)"
	@echo
	@echo "Link Flags:"
	@echo "  Link flags: $(LFLAGS)"
	@echo

# Cleanup anything this makefile can create
clean:
	@[ -f ./$(TESTNAME) ] && rm ./$(TESTNAME) ; true
	@[ -f ./obin ] && rm ./obin ; true
	@[ -f ./$(TESTNAME).ll ] && rm *.ll ; true
	@[ -f ./$(TESTNAME).o ] && rm $(TESTNAME).o ; true
	@[ -f ./$(TESTNAME).s ] && rm *.s ; true
