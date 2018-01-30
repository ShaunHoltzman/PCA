#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <mpi.h>

void *pgm_read(const char *p)
{
	FILE *file = fopen(p, "r");
	if (!file)
		return 0;

	size_t w, h;
	void *buf = 0;
	if (fscanf(file, "P5\n%zu %zu\n255\n", &w, &h) < 2)
		goto err;

	if (!w || !h || w > SIZE_MAX/h)
		goto err;

	if (!(buf = malloc(h*w)))
		goto err;

	if (fread(buf, 1, h*w, file) != h*w)
		goto err;

	fclose(file);
	return buf;
err:
	free(buf);
	fclose(file);
	return 0;
}

int pgm_write(const char *p, void *buf, size_t width, size_t height)
{
	FILE *file = fopen(p, "wb");
	if (!file)
		return -1;

	fprintf(file, "P5\n%zu %zu\n255\n", width, height);
	fwrite(buf, 1, height*width, file);
	fclose(file);
	return 0;
}

int main(int argc, char **argv)
{
	if (argc != 5) {
		fprintf(stderr, "sobel IN OUT W H\n");
		return 1;
	}
	const char *in = argv[1];
	const char *out = argv[2];
	int W = atoi(argv[3]);
	int H = atoi(argv[4]);

	int num_tasks, rank;
	
	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &num_tasks);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	
	int *buf;
	unsigned char *pgm;

	// Initialize
	if(rank==0)
	{
		buf = malloc(W * H * sizeof *buf);
		pgm = pgm_read(in);
		if (!pgm) {
			fprintf(stderr, "failed to read %s\n", in);
			return 1;
		}
		for (size_t i=0; i < W*H; i++)
			buf[i] = pgm[i];
	}
		
	int H_new;
	int remainder;
	
	H_new = floor(H/num_tasks);

	remainder = H-H_new*num_tasks;
			
	int *send_count = malloc(sizeof(int)*num_tasks);
	int *disp_count = malloc(sizeof(int)*num_tasks);
	
	for(int i=0; i<num_tasks; i++)
	{
		if(i==0)
		{
			send_count[i] = W*(H_new+1);
			disp_count[i] = 0;
		}
		else if(i==num_tasks-1)
		{
			send_count[i] = W*(H_new+1+remainder);
			disp_count[i] = W*H_new*i - W;
		}
		else
		{
			send_count[i] = W*(H_new+2);
			disp_count[i] = W*H_new*i - W;
		}	
	}
	
	int recv_size;
	int *recv_buf;
	int *gx;
	int *gy;
	
	if(rank == 0)
	{
		recv_size = W*(H_new+1);
		recv_buf = malloc(recv_size*sizeof(int));
		gx = malloc(W * H_new * sizeof *gx);
		gy = malloc(W * H_new * sizeof *gy);
	}
	else if(rank==num_tasks-1)
	{
		recv_size = W*(H_new+1+remainder);
		recv_buf = malloc(recv_size*sizeof(int));
		gx = malloc(W * (H_new+remainder) * sizeof *gx);
		gy = malloc(W * (H_new+remainder) * sizeof *gy);
	}
	else
	{
		recv_size = W*(H_new+2);
		recv_buf = malloc(recv_size*sizeof(int));
		gx = malloc(W * (H_new+1) * sizeof *gx);
		gy = malloc(W * (H_new+1) * sizeof *gy);
	}
			
	MPI_Scatterv(buf, send_count, disp_count, MPI_INT, recv_buf, recv_size, MPI_INT, 0, MPI_COMM_WORLD);
	
	if(rank==0)
	{
		for (size_t i = 1; i < H_new; i++)
			for (size_t j = 1; j < W-1; j++)
				gx[i*W + j] =
					1*recv_buf[(i-1)*W + (j-1)] + 0*recv_buf[(i-1)*W + j] + -1*recv_buf[(i-1)*W + (j+1)] +
					2*recv_buf[(i+0)*W + (j-1)] + 0*recv_buf[(i+0)*W + j] + -2*recv_buf[(i+0)*W + (j+1)] +
					1*recv_buf[(i+1)*W + (j-1)] + 0*recv_buf[(i+1)*W + j] + -1*recv_buf[(i+1)*W + (j+1)];
		for (size_t i = 1; i < H_new; i++)
			for (size_t j = 1; j < W-1; j++)
				gy[i*W + j] =
					 1*recv_buf[(i-1)*W + (j-1)] +  2*recv_buf[(i-1)*W + j] +  1*recv_buf[(i-1)*W + (j+1)] +
					 0*recv_buf[(i+0)*W + (j-1)] +  0*recv_buf[(i+0)*W + j] +  0*recv_buf[(i+0)*W + (j+1)] +
					-1*recv_buf[(i+1)*W + (j-1)] + -2*recv_buf[(i+1)*W + j] + -1*recv_buf[(i+1)*W + (j+1)];
		for (size_t i = 1; i < H_new; i++)
			for (size_t j = 1; j < W-1; j++) {
				int x = gx[i*W + j];
				int y = gy[i*W + j];
				recv_buf[i*W + j] = sqrt(x*x + y*y);
			}
	}
	else if(rank==num_tasks-1)
	{
		for (size_t i = 1; i < H_new+remainder; i++)
			for (size_t j = 1; j < W-1; j++)
				gx[i*W + j] =
					1*recv_buf[(i-1)*W + (j-1)] + 0*recv_buf[(i-1)*W + j] + -1*recv_buf[(i-1)*W + (j+1)] +
					2*recv_buf[(i+0)*W + (j-1)] + 0*recv_buf[(i+0)*W + j] + -2*recv_buf[(i+0)*W + (j+1)] +
					1*recv_buf[(i+1)*W + (j-1)] + 0*recv_buf[(i+1)*W + j] + -1*recv_buf[(i+1)*W + (j+1)];
		for (size_t i = 1; i < H_new+remainder; i++)
			for (size_t j = 1; j < W-1; j++)
				gy[i*W + j] =
					 1*recv_buf[(i-1)*W + (j-1)] +  2*recv_buf[(i-1)*W + j] +  1*recv_buf[(i-1)*W + (j+1)] +
					 0*recv_buf[(i+0)*W + (j-1)] +  0*recv_buf[(i+0)*W + j] +  0*recv_buf[(i+0)*W + (j+1)] +
					-1*recv_buf[(i+1)*W + (j-1)] + -2*recv_buf[(i+1)*W + j] + -1*recv_buf[(i+1)*W + (j+1)];
		for (size_t i = 1; i < H_new+remainder; i++)
			for (size_t j = 1; j < W-1; j++) {
				int x = gx[i*W + j];
				int y = gy[i*W + j];
				recv_buf[i*W + j] = sqrt(x*x + y*y);
			}
	}
	else
	{
		for (size_t i = 1; i < H_new+1; i++)
			for (size_t j = 1; j < W-1; j++)
				gx[i*W + j] =
					1*recv_buf[(i-1)*W + (j-1)] + 0*recv_buf[(i-1)*W + j] + -1*recv_buf[(i-1)*W + (j+1)] +
					2*recv_buf[(i+0)*W + (j-1)] + 0*recv_buf[(i+0)*W + j] + -2*recv_buf[(i+0)*W + (j+1)] +
					1*recv_buf[(i+1)*W + (j-1)] + 0*recv_buf[(i+1)*W + j] + -1*recv_buf[(i+1)*W + (j+1)];
		for (size_t i = 1; i < H_new+1; i++)
			for (size_t j = 1; j < W-1; j++)
				gy[i*W + j] =
					 1*recv_buf[(i-1)*W + (j-1)] +  2*recv_buf[(i-1)*W + j] +  1*recv_buf[(i-1)*W + (j+1)] +
					 0*recv_buf[(i+0)*W + (j-1)] +  0*recv_buf[(i+0)*W + j] +  0*recv_buf[(i+0)*W + (j+1)] +
					-1*recv_buf[(i+1)*W + (j-1)] + -2*recv_buf[(i+1)*W + j] + -1*recv_buf[(i+1)*W + (j+1)];
		for (size_t i = 1; i < H_new+1; i++)
			for (size_t j = 1; j < W-1; j++) {
				int x = gx[i*W + j];
				int y = gy[i*W + j];
				recv_buf[i*W + j] = sqrt(x*x + y*y);
			}
	}
	
	int *recv_count = malloc(sizeof(int)*num_tasks);
	int *disp_count_r = malloc(sizeof(int)*num_tasks);
	int index;
	int size;
	
	for(int i=0; i<num_tasks; i++)
	{
		
		if(i == 0)
		{
			recv_count[i] = W*(H_new);
			disp_count_r[i] = 0;
		}
		else if(i == num_tasks-1)
		{
			recv_count[i] = W*(H_new+remainder);
			disp_count_r[i] = W*(H_new)*i;
		}
		else
		{
			recv_count[i] = W*(H_new);
			disp_count_r[i] = W*(H_new)*i;
		}	
		
	}
	
	if(rank == 0)
		index=0;
	else
		index=W;
	
	if(rank==num_tasks-1)
		size = W*(H_new+remainder);
	else
		size = W*(H_new);

	MPI_Gatherv(&recv_buf[index], size, MPI_INT, buf, recv_count, disp_count_r, MPI_INT, 0, MPI_COMM_WORLD);
	
	// Finalize
	if(rank==0)
	{
		for (size_t i = 0; i < W*H; i++)
			pgm[i] = buf[i];
		pgm_write(out, pgm, W, H);
		free(pgm);
	}
	
	MPI_Finalize();
	
}
