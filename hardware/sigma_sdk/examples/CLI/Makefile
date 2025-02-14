#  Copyright (C) 2001-2020 Force Dimension
#  All Rights Reserved.
#
#  Version 3.9.1


# import settings
TOP_DIR = ../..
include $(TOP_DIR)/Makefile.common

# haptic examples
SUBDIRS += encoders \
           jacobian \
           gravity \
           multithread \
           segment \
           viscosity \
           wrist

# robotic examples
SUBDIRS += autocenter \
           autoinit \
           center \
           mirror \
           robot \
           sensor

all: $(SUBDIRS)

.PHONY: $(SUBDIRS)
$(SUBDIRS):
	$(MAKE) -C $@

clean:
	@for T in $(SUBDIRS); do make -C $$T $@; done
