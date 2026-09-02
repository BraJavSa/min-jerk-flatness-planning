CC = gcc
CFLAGS = -O3 -march=native -Wall -Wextra -std=c11 -Iinclude
LDFLAGS = -lm

SRC_COMMON = src/c2_usv_params.c src/c2_min_jerk_qp.c src/c2_sqp_optimizer.c src/c2_flatness_reconstruct.c

all: c2_main c2_simulate_openloop

c2_main: $(SRC_COMMON) src/c2_main.c
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

c2_simulate_openloop: $(SRC_COMMON) src/c2_simulate_openloop.c
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

run: all
	./c2_main
	./c2_simulate_openloop

plots: run
	python3 plot_results.py

bench: c2_main
	@echo "Running c2_main 20x to measure steady-state timing..."
	@for i in $$(seq 1 20); do ./c2_main | grep "Compute Time"; done

clean:
	rm -f c2_main c2_simulate_openloop *.csv *.json *.png

.PHONY: all run plots bench clean
