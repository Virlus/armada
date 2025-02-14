#  Copyright (C) 2001-2020 Force Dimension
#  All Rights Reserved.
#
#  Version 3.9.1


# import settings
TOP_DIR = ../..
include $(TOP_DIR)/Makefile.common

# haptic examples
SUBDIRS += constraints \
           sphere \
           torus \
           torques \
           transformations

all: $(SUBDIRS)

$(SUBDIRS): glfw
	$(MAKE) -C $@

glfw:
	$(MAKE) -C $(GLFW_DIR)

.PHONY: $(SUBDIRS) glfw

clean:
	@for T in $(SUBDIRS); do make -C $$T $@; done
