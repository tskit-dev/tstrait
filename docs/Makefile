# Need to set PYTHONPATH so that we pick up the local tstrait
PYPATH=${PWD}/..
MSP_VERSION:=$(shell PYTHONPATH=${PYPATH} \
   python3 -c 'import tstrait')

dev:
	PYTHONPATH=${PYPATH} ./build.sh

clean:
	rm -fR _build

## tskit has some sentences regarding dOxygen. Is it necessary?

#DOXYGEN_XML=doxygen/xml
#
#all: ${DOXYGEN_XML} dev
#
#${DOXYGEN_XML}: ../c/tskit/*.h
#	cd doxygen && doxygen
#
#DOXYGEN_XML=doxygen/xml
#
#all: ${DOXYGEN_XML} dev
#
#${DOXYGEN_XML}: ../c/tskit/*.h
#	cd doxygen && doxygen