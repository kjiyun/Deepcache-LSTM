CLIENT_FILES=simulator/client.cpp
SERVER_FILES=simulator/app.cpp
SERVER_COMPILE_ARGS=-std=c++17 -pthread -lcurl -I$(shell pwd)/simulator/include

HR_FILES=hr/simulator.cpp hr/hr.cpp hr/cache.cpp hr/requests.cpp hr/model.cpp hr/utils.cpp hr/metadata.cpp
HR_COMPILE_ARGS=-std=c++17 -pthread -I$(shell pwd)/include -Llibs -l_lightgbm -Wl,-rpath,$(shell pwd)/libs
HR_OPTIMIZATION_ARGS=-O3 -funroll-loops -flto

HR_LIB=libs/liblfh.a
HR_LIB_COMPILE_ARGS=-std=c++17 -pthread -I$(shell pwd)/include

.PHONY: all hr prepare_lib move_lib build_lightgbm debug app client build_ats dev_ats

all: hr app client

build_lightgbm:
	if [ -f "libs/lib_lightgbm.so" ]; then \
		echo "LightGBM Already Installed!"; \
	else \
		if [ -d "LightGBM" ]; then \
			cd LightGBM && git pull && cd ..; \
		else \
			git clone --recursive https://github.com/Microsoft/LightGBM; \
		fi && \
		cd LightGBM && \
		mkdir -p build && \
		cd build && \
		cmake .. && \
		make -j1 && \
		cd ../.. && \
		mkdir -p libs && \
		cp LightGBM/lib_lightgbm.dylib libs/ && \
		rm -rf LightGBM; \
	fi

hr: $(HR_FILES)
	g++ -o executables/hr $(HR_FILES) $(HR_COMPILE_ARGS) $(HR_OPTIMIZATION_ARGS)

prepare_lib: $(HR_FILES)
	@mkdir -p libs
	@for file in $(HR_FILES); do \
		g++ -c -arch x86_64 $$file $(HR_LIB_COMPILE_ARGS) $(HR_OPTIMIZATION_ARGS) -o `basename $$file .cpp`-x86_64.o; \
		g++ -c -arch arm64 $$file $(HR_LIB_COMPILE_ARGS) $(HR_OPTIMIZATION_ARGS) -o `basename $$file .cpp`-arm64.o; \
	done
	@ar rvs libs/liblfh-x86_64.a *-x86_64.o
	@ar rvs libs/liblfh-arm64.a *-arm64.o
	@lipo -create -output $(HR_LIB) libs/liblfh-x86_64.a libs/liblfh-arm64.a
	@rm *.o libs/liblfh-x86_64.a libs/liblfh-arm64.a

debug: $(HR_FILES)
	g++ -o executables/debug_hr $(HR_FILES) $(HR_COMPILE_ARGS) -g -fsanitize=address

app: $(SERVER_FILES)
	g++ -o executables/app $(SERVER_FILES) $(SERVER_COMPILE_ARGS) $(HR_OPTIMIZATION_ARGS)

client: $(CLIENT_FILES)
	g++ -o executables/client $(CLIENT_FILES) $(SERVER_COMPILE_ARGS) $(HR_OPTIMIZATION_ARGS)

prepare_ats: prepare_lib
	@mkdir -p trafficserver/iocore/cache/hr/libs
	@cp -r libs/* trafficserver/iocore/cache/hr/libs
	@mkdir -p trafficserver/iocore/cache/hr/include
	@cp -r include/* trafficserver/iocore/cache/hr/include

build_ats: prepare_ats
	cd trafficserver && \
	autoreconf -if && \
	./configure --prefix=$(shell pwd)/ats && \
	make clean && \
	make && \
	make install

dev_ats: prepare_ats
	cd trafficserver && \
	make && \
	make install && \
	cd ../ && \
	cp -r trafficserver_configs/* ats/etc/trafficserver/ && \
	ats/bin/traffic_server -K -k -T http_seq