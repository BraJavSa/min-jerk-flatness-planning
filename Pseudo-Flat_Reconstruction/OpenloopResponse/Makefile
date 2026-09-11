CC = gcc
CFLAGS = -Wall -Wextra -O2 -std=c99 -D_POSIX_C_SOURCE=199309L
LDLIBS = -lm

COMMON_OBJS = usv_params.o min_jerk_qp.o sqp_optimizer.o flatness_reconstruct.o

all: main simulate_openloop

main: main.o $(COMMON_OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)

simulate_openloop: simulate_openloop.o $(COMMON_OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)

%.o: %.c
	$(CC) $(CFLAGS) -c -o $@ $<

usv_params.o: usv_params.c usv_params.h
min_jerk_qp.o: min_jerk_qp.c min_jerk_qp.h
sqp_optimizer.o: sqp_optimizer.c sqp_optimizer.h flatness_reconstruct.h
flatness_reconstruct.o: flatness_reconstruct.c flatness_reconstruct.h usv_params.h sqp_optimizer.h
main.o: main.c usv_params.h min_jerk_qp.h flatness_reconstruct.h
simulate_openloop.o: simulate_openloop.c usv_params.h min_jerk_qp.h flatness_reconstruct.h

clean:
	rm -f *.o main simulate_openloop *.csv *.json *.png

.PHONY: all clean
