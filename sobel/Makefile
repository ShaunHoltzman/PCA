CC := mpicc
CFLAGS += -std=c99
LDFLAGS += -lm

sobel: sobel.c
	$(CC) -o $@ $(CFLAGS) $< $(LDFLAGS)

clean:
	$(RM) -f sobel

.PHONY: clean
