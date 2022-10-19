/** 
 * Read .mtx input file and convert to CSR format
 * Args: {fileName1, fileName2, mat->ptr, mat->idx, mat->val flags, #rows, #elements}
 * Read whole matrix and sort with from data.
 * Divide Matrix A to small blocks based of number of elements.
 * Convert each block to CSR format then let main process to compute them.
 */

#include "SSpMM.hpp"

struct edgeData *queues[10];
int queue_Offsets[10];

void make_CSR(std::string input_File, csr *mat);
void sort_From(edgeData *edges, int nr, int ne);


void read_CSR(std::string input_File, csr *mat) {
    if (access((input_File + ".csr").c_str(), F_OK) < 0) {
        printf("Make %s\n", (input_File + ".csr").c_str());
        make_CSR(input_File, mat);

        return;
    }

    FILE *fp;
    fp = fopen((input_File + ".csr").c_str(), "rb");
    int row, col, edge;
    fread(&row, sizeof(int), 1, fp);
    fread(&col, sizeof(int), 1, fp);
    fread(&edge, sizeof(int), 1, fp);
    mat->nr = row;
    mat->nc = col;
    mat->ne = edge;

    mat->ptr = (int*)malloc(sizeof(int) * (mat->nr + 1));
    mat->idx = (int*)malloc(sizeof(int) * mat->ne);
    mat->val = (float*)malloc(sizeof(float) * mat->ne);

    fread(mat->ptr, sizeof(int), mat->nr + 1, fp);
    fread(mat->idx, sizeof(int), mat->ne, fp);
    fread(mat->val, sizeof(float), mat->ne, fp);
    fclose(fp);

    printf("Input: %s\n\tNumber of Rows:\t\t%d\n\tNumber of Columns:\t%d\n\tNumber of Edges:\t%d\n\n", input_File.c_str(), mat->nr, mat->nc, mat->ne);
}

void make_CSR(std::string input_File, csr *mat) {
    FILE *fp = fopen(input_File.c_str(), "rt");

    char line[200];
    bool symmetric = false;
    if (strstr(input_File.c_str(), ".dat") == NULL) {
        fgets(line, 200, fp);
        if(strstr(line, "symmetric")) symmetric = true;
    }
    //else puts("Synthetic");

    while (fgets(line, 200, fp) != NULL) {
        if (line[0] == '%' || line[0] == '#') continue;   // Comment
        char delims[] = "\t, ";
        char *t;
        t = strtok(line, delims);
        if (t == NULL) {
            fprintf(stderr, "Wrong Format.\n");
            exit(-1);
        }
        mat->nr = atoi(t) + 1;
        
        t = strtok(NULL, delims);
        if (t == NULL) {
            fprintf(stderr, "Wrong Format.\n");
            exit(-1);
        }
        mat->nc = atoi(t) + 1;

        t = strtok(NULL, delims);
        if (t == NULL) {
            fprintf(stderr, "Wrong Format.\n");
            exit(-1);
        }
        mat->ne = atoi(t);
        if (symmetric == true) mat->ne *= 2;

        break;
    }
    
    mat->ptr = (int*)malloc(sizeof(int) * (mat->nr + 1));
    mat->idx = (int*)malloc(sizeof(int) * mat->ne);
    mat->val = (float*)malloc(sizeof(float) * mat->ne);

    edgeData *edges = (edgeData*)malloc(sizeof(edgeData) * mat->ne);
    for (int q = 0; q < 10; ++q) queues[q] = (edgeData*)malloc(sizeof(edgeData) * mat->ne);
    int offset = 0;

    while (fgets(line, 200, fp) != NULL) {
        char delims[] = "\t, ";
        char *t;
        t = strtok(line, delims);
        //if (t == NULL) {
        //    fprintf(stderr, "Wrong Format.\n");
        //    exit(-1);
        //}
        edges[offset].from = atoi(t);
        t = strtok(NULL, delims);
        //if (t == NULL) {
        //    fprintf(stderr, "Wrong Format.\n");
        //    exit(-1);
        //}
        edges[offset].to = atoi(t);

        t = strtok(NULL, delims);
        if (t != NULL) edges[offset].val = atof(t);
        else edges[offset].val = 1.0f;

        if ((symmetric == true) && (edges[offset].from != edges[offset].to)) {
            edges[offset + 1].from = edges[offset].to;
            edges[offset + 1].to = edges[offset].from;
            edges[offset + 1].val = edges[offset].val;

            offset++;
        }
        offset++;
    }
    mat->ne = offset;

    sort_From(edges, mat->nr, mat->ne);
    
    for (int q = 0; q < 10; ++q) {
        free(queues[q]);
    }

    mat->ptr[0] = 0;
    mat->ptr[mat->nr] = mat->ne;

    int row_Offset = 0;
    int edges_in_row = 0;
    for (int e = 0; e < mat->ne; ++e) {
        if (edges[e].from > row_Offset) {
            mat->ptr[row_Offset + 1] = mat->ptr[row_Offset] + edges_in_row;

            for (int empty_Rows = row_Offset + 1; empty_Rows < edges[e].from; ++empty_Rows) {
                mat->ptr[empty_Rows + 1] = mat->ptr[empty_Rows];
            }

            row_Offset = edges[e].from;
            edges_in_row = 0;
        }
        mat->idx[e] = edges[e].to;
        mat->val[e] = edges[e].val;
        edges_in_row++;
    }

    free(edges);
    fclose(fp);

    printf("Input: %s\n\tNumber of Rows:\t\t%d\n\tNumber of Columns:\t%d\n\tNumber of Edges:\t%d\n\n", input_File.c_str(), mat->nr, mat->nc, mat->ne);

/*    int fd = open((input_File + ".csr").c_str(), O_CREAT | O_WRONLY, 0644);
    write(fd, &row, sizeof(int));
    write(fd, &col, sizeof(int));
    write(fd, &edge, sizeof(int));

    close(fd);*/

    fp = fopen((input_File + ".csr").c_str(), "wb");
    int row = mat->nr, col = mat->nc, edge = mat->ne;
    fwrite(&row, sizeof(int), 1, fp);
    fwrite(&col, sizeof(int), 1, fp);
    fwrite(&edge, sizeof(int), 1, fp);

    fwrite(mat->ptr, sizeof(int), row + 1, fp);
    fwrite(mat->idx, sizeof(int), edge, fp);
    fwrite(mat->val, sizeof(float), edge, fp);
    fclose(fp);
}


/** 
 * Radix Sort
 * Each edge has {from, to, value} data
 * Sort only with from data
 * Total Memory Usage = sizeof(struct edgeData) * mat->ne +                                               : Temporary Array(edges in main Function) to read input file
 *                      10 * sizeof(struct edgeData) * mat->ne +                                                      : Queue Size
 *                      sizeof(int) * mat->nr + sizeof(int) * mat->ne + sizeof(float) * mat->ne        : Output Array(CSR Format)
 *                      Actually, Ouput Array Size is hidden since memory allocated after freeing Queues
 * Can reduce memory usage by changing implemention to 2-bit radix sort
 * -> Queue Size = 2 * sizeof(int) * mat->ne
 * But, time complexity increases.
 */

void sort_From(struct edgeData *edges, int nr, int ne) {
    int num_Digits = 1;
    while (nr) {
        nr /= 10;
        num_Digits *= 10;
    }
    //printf("Number of Digits: %d\n", num_Digits);

    for (int divide_To = 1; divide_To <= num_Digits; divide_To *= 10) {
        for (int q = 0; q < 10; ++q) queue_Offsets[q] = 0;
        
        for (int e = 0; e < ne; ++e) {
            int row = edges[e].from;
            int target_Queue = row % divide_To;
            target_Queue = (target_Queue * 10) / divide_To;
            queues[target_Queue][queue_Offsets[target_Queue]++] = edges[e];
        }

        int offset = 0;
        for (int q = 0; q < 10; ++q) {
            memcpy(edges + offset, queues[q], sizeof(struct edgeData) * queue_Offsets[q]);
            offset += queue_Offsets[q];
        }
    }
}