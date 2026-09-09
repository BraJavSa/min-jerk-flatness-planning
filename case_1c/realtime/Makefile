CC = gcc
CFLAGS = -O3 -march=native -ffast-math -Wall
LDLIBS = -lm

COMMON_OBJS = usv_params.o min_jerk_qp.o flatness_reconstruct.o

all: main

main: main.o $(COMMON_OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)

%.o: %.c
	$(CC) $(CFLAGS) -c -o $@ $<

clean:
	rm -f *.o main

.PHONY: all clean
