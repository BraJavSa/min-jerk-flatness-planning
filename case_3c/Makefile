CC = gcc
IPOPT_INC = /home/brayan/.local/lib/python3.12/site-packages/casadi/include
IPOPT_LIB = /home/brayan/.local/lib/python3.12/site-packages/casadi

CFLAGS = -O3 -march=native -ffast-math -Wall -Wno-deprecated-declarations -I$(IPOPT_INC)
LDFLAGS = -L$(IPOPT_LIB) -Wl,-rpath,$(IPOPT_LIB)
LDLIBS = -lipopt -lm

COMMON_OBJS = usv_params.o trajectory_nlp.o flatness_reconstruct.o

all: main simulate_openloop

main: main.o $(COMMON_OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS) $(LDLIBS)

simulate_openloop: simulate_openloop.o $(COMMON_OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS) $(LDLIBS)

%.o: %.c
	$(CC) $(CFLAGS) -c -o $@ $<

clean:
	rm -f *.o main simulate_openloop

.PHONY: all clean
