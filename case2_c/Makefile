CC = gcc
CFLAGS = -O2 -Wall -Wextra -std=c11 -D_POSIX_C_SOURCE=199309L -Iinclude
LDFLAGS = -llapacke -llapack -lblas -lm

SRC_COMMON = src/usv_params.c src/min_jerk_qp.c src/flatness_reconstruct.c
OBJ_COMMON = $(SRC_COMMON:.c=.o)

all: c2_main c2_simulate_openloop

c2_main: src/main.o $(OBJ_COMMON)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

c2_simulate_openloop: src/simulate_openloop.o $(OBJ_COMMON)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

%.o: %.c
	$(CC) $(CFLAGS) -c -o $@ $<

clean:
	rm -f src/*.o c2_main c2_simulate_openloop *.csv *.json

.PHONY: all clean
