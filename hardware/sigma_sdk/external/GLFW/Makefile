#  Copyright (C) 2001-2020 Force Dimension
#  All Rights Reserved.
#
#  Makefile for GLFW Version 3.3.2


# top level directory
TOP_DIR = .

# determine build configuration
ifneq ($(CFG),debug)
CFG = release
endif

# determine platform
ARCH   ?= $(shell echo `uname -m`)
SYSTEM ?= $(shell echo `uname -s`)
ifeq ($(SYSTEM), Linux)
OS       = lin
CC      ?= gcc
COMPILER = $(CC)
endif
ifeq ($(SYSTEM), Darwin)
OS       = mac
CC      ?= clang
COMPILER = $(CC)
endif

# paths
LIB_DIR = $(TOP_DIR)/lib/$(CFG)/$(OS)-$(ARCH)-$(COMPILER)
INC_DIR = $(TOP_DIR)/include
SRC_DIR = $(TOP_DIR)/src
OBJ_DIR = $(TOP_DIR)/obj/$(CFG)/$(OS)-$(ARCH)-$(COMPILER)

# librarian flags
ARFLAGS  = rs

# build configuration specific flags
ifneq ($(CFG),debug)
CFLAGS += -O3 -w
else
CFLAGS += -O0 -g -DDEBUG
endif

# set common source files
CSOURCES += $(SRC_DIR)/context.c \
            $(SRC_DIR)/init.c \
            $(SRC_DIR)/input.c \
            $(SRC_DIR)/monitor.c \
            $(SRC_DIR)/vulkan.c \
            $(SRC_DIR)/window.c

# set target architecture
ifeq ($(OS), lin)
CFLAGS   += -DLINUX -D_GLFW_X11
CSOURCES += $(SRC_DIR)/x11_init.c \
            $(SRC_DIR)/x11_monitor.c \
            $(SRC_DIR)/x11_window.c \
            $(SRC_DIR)/xkb_unicode.c \
            $(SRC_DIR)/posix_time.c \
            $(SRC_DIR)/posix_thread.c \
            $(SRC_DIR)/glx_context.c \
            $(SRC_DIR)/egl_context.c \
            $(SRC_DIR)/osmesa_context.c \
            $(SRC_DIR)/linux_joystick.c
ifeq ($(ARCH), x86_64)
CFLAGS   += -m64 -march=x86-64
endif
ifneq (,$(filter $(ARCH),i386 i486 i586 i686))
CFLAGS   += -m32 -march=$(ARCH) -msse2 -mtune=generic
endif
ifeq ($(ARCH), native)
CFLAGS   += -march=native
endif
endif
ifeq ($(OS), mac)
CFLAGS   += -DMACOSX -D_GLFW_COCOA -arch $(ARCH)
CSOURCES += $(SRC_DIR)/cocoa_init.m \
            $(SRC_DIR)/cocoa_joystick.m \
            $(SRC_DIR)/cocoa_monitor.m \
            $(SRC_DIR)/cocoa_window.m \
            $(SRC_DIR)/cocoa_time.c \
            $(SRC_DIR)/posix_thread.c \
            $(SRC_DIR)/nsgl_context.m \
            $(SRC_DIR)/egl_context.c \
            $(SRC_DIR)/osmesa_context.c
endif

# target
LIB_TARGET = $(LIB_DIR)/libglfw.a

# source files
CFLAGS    += -I$(INC_DIR) -I$(SRC_DIR)
OBJECTS    = $(patsubst %.c, $(OBJ_DIR)/%.o, $(CSOURCES))
OBJECTS   += $(patsubst %.m, $(OBJ_DIR)/%.o, $(CSOURCES))
OBJ_TREE   = $(sort $(dir $(OBJECTS)))
INCLUDES   = $(wildcard $(INC_DIR)/GLFW/*.h)
VPATH     += $(SRC_DIR)

# target rules

all: lib

lib: $(LIB_TARGET)

$(LIB_TARGET): $(OBJECTS) | $(LIB_DIR)
	$(AR) $(ARFLAGS) $(LIB_TARGET) $?

$(OBJECTS): $(INCLUDES) | $(OBJ_TREE)

$(LIB_DIR) $(OBJ_TREE):
	mkdir -p $@

$(OBJ_DIR)/%.o : %.c
	$(CC) $(CFLAGS) -c -o $@ $<

$(OBJ_DIR)/%.o : %.m
	$(CC) $(CFLAGS) -c -o $@ $<

tags:
	find ../.. -name \*.cpp -o -name \*h | xargs etags -o TAGS

clean:
	-rm -f $(LIB_TARGET) *~
	-rm -rf $(OBJ_DIR)

