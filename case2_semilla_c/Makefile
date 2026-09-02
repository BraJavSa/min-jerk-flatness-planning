CC = gcc
CFLAGS = -O3 -march=native -Wall -Wextra -std=c11
LDFLAGS = -lm

TARGET = c2_flatness
SRC = c2_flatness.c

all: $(TARGET)

$(TARGET): $(SRC)
	$(CC) $(CFLAGS) -o $(TARGET) $(SRC) $(LDFLAGS)

run: $(TARGET)
	./$(TARGET)

bench: $(TARGET)
	@echo "Running 20x to measure steady-state timing..."
	@for i in $$(seq 1 20); do ./$(TARGET) | grep "Total pipeline"; done

clean:
	rm -f $(TARGET) c2_flatness_c_output.csv

.PHONY: all run bench clean
