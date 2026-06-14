/*
 *  随机数的统计分布测试 & 蒙特卡罗法求π
 *
 *  功能：
 *  1. 使用 rand()%N 生成随机数，统计频率与重复间隔
 *  2. 六种随机性检验：卡方、K-S、序列、游程、自相关、累加和
 *  3. 设计并测试均匀、正态分布随机数
 *  4. 蒙特卡罗法求 π
 *  5. 将原始序列导出为二进制文件
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

/* 确保 M_PI 可用 */
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define N           100         // 随机数范围 [0, N-1]
#define SAMPLES     100000      // 主测试序列长度
#define PI_SAMPLES  1000000     // 求 π 的投点数量

/* 比较函数，用于 qsort */
int cmp_int(const void *a, const void *b) {
    return (*(int*)a - *(int*)b);
}

/* 
  基础随机数生成函数
*/

/* 原始方法：rand() % N */
int rand_mod_N(void) {
    return rand() % N;
}

/* 改进的均匀整数 [0, max-1]，避免低位偏差 */
int uniform_int(int max) {
    return (int)((rand() / (RAND_MAX + 1.0)) * max);
}

/* 均匀浮点数 [0, 1) */
double uniform_double(void) {
    return rand() / (RAND_MAX + 1.0);
}

/* 标准正态分布 N(0,1) —— Box-Muller 变换 */
double gaussian_box_muller(void) {
    double u1 = uniform_double();
    double u2 = uniform_double();
    double z = sqrt(-2.0 * log(u1 + 1e-30)) * cos(2.0 * M_PI * u2);
    return z;
}

/* 一般正态分布 N(mu, sigma) */
double gaussian(double mu, double sigma) {
    return mu + sigma * gaussian_box_muller();
}

/* 
 统计与间隔分析
 */

/* 打印频率分布 */
void print_frequency(const int *data, int len, int range) {
    int *counts = calloc(range, sizeof(int));
    for (int i = 0; i < len; i++) counts[data[i]]++;

    printf("%-6s %-8s %-12s %-12s\n", "Value", "Count", "Prob", "TheoProb");
    printf("------ -------- ------------ ------------\n");
    for (int i = 0; i < range; i++) {
        printf("%-6d %-8d %-12.5f %-12.5f\n",
               i, counts[i],
               (double)counts[i] / len,
               1.0 / range);
    }
    free(counts);
}

/* 统计并打印重复间隔分布 */
void gap_distribution(const int *data, int len, int range) {
    int *last_pos = malloc(range * sizeof(int));
    for (int i = 0; i < range; i++) last_pos[i] = -1;

    int *gaps = malloc(len * sizeof(int));
    int gap_count = 0;

    for (int i = 0; i < len; i++) {
        int v = data[i];
        if (last_pos[v] != -1) {
            gaps[gap_count++] = i - last_pos[v];
        }
        last_pos[v] = i;
    }

    if (gap_count == 0) {
        printf("没有发现重复，无法统计间隔。\n");
        free(last_pos); free(gaps);
        return;
    }

    int max_gap = 0;
    for (int i = 0; i < gap_count; i++)
        if (gaps[i] > max_gap) max_gap = gaps[i];

    int nbins = 20;
    int bin_width = (max_gap + nbins - 1) / nbins;
    if (bin_width < 1) bin_width = 1;

    int *hist = calloc(nbins, sizeof(int));
    for (int i = 0; i < gap_count; i++) {
        int idx = gaps[i] / bin_width;
        if (idx >= nbins) idx = nbins - 1;
        hist[idx]++;
    }

    printf("\n---------- 重复间隔分布 ----------\n");
    printf("总间隔数: %d, 最大间隔: %d, 区间宽度: %d\n", gap_count, max_gap, bin_width);
    printf("%-16s %-8s\n", "Gap Interval", "Count");
    printf("---------------- --------\n");
    for (int i = 0; i < nbins; i++) {
        int low = i * bin_width;
        int high = (i == nbins - 1) ? max_gap : (i + 1) * bin_width - 1;
        printf("[%-6d, %-6d] %-8d\n", low, high, hist[i]);
    }

    free(last_pos); free(gaps); free(hist);
}

/*
 六种随机性检验
 */

/* 1. 卡方拟合优度检验 */
double test_chi_square(const int *data, int len, int range) {
    int *counts = calloc(range, sizeof(int));
    for (int i = 0; i < len; i++) counts[data[i]]++;
    double expected = (double)len / range;
    double chi2 = 0.0;
    for (int i = 0; i < range; i++) {
        double diff = counts[i] - expected;
        chi2 += diff * diff / expected;
    }
    free(counts);
    return chi2;
}

/* 2. K-S 检验 */
double test_ks(const int *data, int len, int range) {
    int *sorted = malloc(len * sizeof(int));
    memcpy(sorted, data, len * sizeof(int));
    qsort(sorted, len, sizeof(int), cmp_int);
    double D = 0.0;
    for (int i = 0; i < len; i++) {
        double Fn = (i + 1.0) / len;
        double F = (sorted[i] + 1.0) / range;
        double diff = fabs(Fn - F);
        if (diff > D) D = diff;
    }
    free(sorted);
    return D;
}

/* 3. 序列检验（二元组卡方） */
void test_serial(const int *data, int len, int range, double *chi2_out) {
    int pairs = range * range;
    int *counts = calloc(pairs, sizeof(int));
    int total = len - 1;
    for (int i = 0; i < total; i++) {
        int idx = data[i] * range + data[i+1];
        counts[idx]++;
    }
    double expected = (double)total / pairs;
    double sum = 0.0;
    for (int i = 0; i < pairs; i++) {
        double diff = counts[i] - expected;
        sum += diff * diff / expected;
    }
    *chi2_out = sum;
    free(counts);
}

/* 4. 游程检验（中位数二值化） */
double test_runs(const int *data, int len, int range) {
    int threshold = range / 2;
    int above = 0, below = 0, runs = 1;
    int current = (data[0] >= threshold) ? 1 : 0;
    (current) ? above++ : below++;
    for (int i = 1; i < len; i++) {
        int bit = (data[i] >= threshold) ? 1 : 0;
        (bit) ? above++ : below++;
        if (bit != current) {
            runs++;
            current = bit;
        }
    }
    double n = len;
    double mean = 2.0 * above * below / n + 1;
    double var = 2.0 * above * below * (2.0 * above * below - n) /
                 (n * n * (n - 1));
    if (var == 0) return 0;
    return (runs - mean) / sqrt(var);
}

/* 5. 自相关检验（滞后1） */
double test_autocorrelation(const int *data, int len, int lag) {
    double mean = 0;
    for (int i = 0; i < len; i++) mean += data[i];
    mean /= len;
    double num = 0, den = 0;
    for (int i = 0; i < len - lag; i++)
        num += (data[i] - mean) * (data[i+lag] - mean);
    for (int i = 0; i < len; i++)
        den += (data[i] - mean) * (data[i] - mean);
    return (den != 0) ? num / den : 0;
}

/* 6. 累加和检验（CUSUM） */
double test_cusum(const int *data, int len, int range) {
    int threshold = range / 2;
    double sum = 0.0, max_abs = 0.0;
    for (int i = 0; i < len; i++) {
        sum += (data[i] >= threshold) ? 1.0 : -1.0;
        if (fabs(sum) > max_abs) max_abs = fabs(sum);
    }
    return max_abs;
}

/*
  蒙特卡罗法求 π
*/
double monte_carlo_pi(int num_points) {
    int inside = 0;
    for (int i = 0; i < num_points; i++) {
        double x = (rand() % 10000) / 1000.0;
        double y = (rand() % 10000) / 1000.0;
        if (x*x + y*y <= 100.0) inside++;
    }
    return 4.0 * inside / num_points;
}

/* 
 * 二进制文件输出
 */

/*
 每数固定1字节写入
 */
void write_bin_fixed_byte(const int *data, int len, const char *filename) {
    FILE *fp = fopen(filename, "wb");
    if (!fp) {
        perror("文件创建失败");
        return;
    }
    for (int i = 0; i < len; i++) {
        unsigned char byte_val = (unsigned char)data[i];
        fwrite(&byte_val, 1, 1, fp);
    }
    fclose(fp);
    printf("已生成二进制文件: %s (%d 字节)\n", filename, len);
}

/*
 *紧凑位打包
 */
void write_bin_packed(const int *data, int len, int bits, const char *filename) {
    FILE *fp = fopen(filename, "wb");
    if (!fp) {
        perror("文件创建失败");
        return;
    }

    unsigned char buffer = 0;
    int bit_count = 0;

    for (int i = 0; i < len; i++) {
        unsigned int value = (unsigned int)data[i];
        for (int b = bits - 1; b >= 0; b--) {
            unsigned char bit = (value >> b) & 1;
            buffer = (buffer << 1) | bit;
            bit_count++;
            if (bit_count == 8) {
                fwrite(&buffer, 1, 1, fp);
                buffer = 0;
                bit_count = 0;
            }
        }
    }

    // 写入末尾不足一个字节的剩余位（低位补0）
    if (bit_count > 0) {
        buffer <<= (8 - bit_count);
        fwrite(&buffer, 1, 1, fp);
    }

    fclose(fp);
    printf("已生成紧凑二进制文件: %s (每样本 %d 比特)\n", filename, bits);
}

int main(void) {
    srand((unsigned)time(NULL));

    /* ---------- 第1部分：生成随机数 ---------- */
    printf("===== 1. 使用 rand()%%%d 生成 %d 个随机数 =====\n\n", N, SAMPLES);
    int *data = malloc(SAMPLES * sizeof(int));
    for (int i = 0; i < SAMPLES; i++) {
        data[i] = rand_mod_N();
    }

    print_frequency(data, SAMPLES, N);
    gap_distribution(data, SAMPLES, N);

    /* ---------- 第2部分：随机性检验 ---------- */
    printf("\n===== 2. 随机性检验结果 =====\n");
    printf("%-24s %-14s\n", "Test", "Statistic");
    printf("------------------------- --------------\n");
    printf("%-24s %14.2f\n", "Chi-square", test_chi_square(data, SAMPLES, N));
    printf("%-24s %14.5f\n", "K-S D", test_ks(data, SAMPLES, N));

    double serial_chi2;
    test_serial(data, SAMPLES, N, &serial_chi2);
    printf("%-24s %14.2f\n", "Serial (pairs)", serial_chi2);

    printf("%-24s %14.3f\n", "Runs Z", test_runs(data, SAMPLES, N));
    printf("%-24s %14.4f\n", "Autocorr (lag=1)", test_autocorrelation(data, SAMPLES, 1));
    printf("%-24s %14.2f\n", "CUSUM max", test_cusum(data, SAMPLES, N));

    /* ---------- 第3部分：改进的分布样本 ---------- */
    printf("\n===== 3. 改进的均匀分布与正态分布样本 =====\n");
    printf("均匀分布 [0, %d) 前20个: ", N);
    for (int i = 0; i < 20; i++) printf("%d ", uniform_int(N));
    printf("\n");

    printf("标准正态分布 N(0,1) 前20个: ");
    for (int i = 0; i < 20; i++) printf("%7.3f", gaussian_box_muller());
    printf("\n");

    printf("正态分布 N(10,3) 前20个: ");
    for (int i = 0; i < 20; i++) printf("%7.3f", gaussian(10.0, 3.0));
    printf("\n");

    /* ---------- 第4部分：蒙特卡罗求 π ---------- */
    printf("\n===== 4. 蒙特卡罗法求 π =====\n");
    double pi_est = monte_carlo_pi(PI_SAMPLES);
    printf("投点次数: %d\n", PI_SAMPLES);
    printf("估计值:   %.8f\n", pi_est);
    printf("真实值:   %.8f\n", M_PI);
    printf("绝对误差: %.8f\n", fabs(pi_est - M_PI));

    /* ---------- 第5部分：导出二进制文件 ---------- */
    printf("\n===== 5. 导出二进制文件供密评工具箱 =====\n");
    // 方法一：固定1字节
    write_bin_fixed_byte(data, SAMPLES, "rand_sequence_byte.bin");
    // 方法二：紧凑7比特
    write_bin_packed(data, SAMPLES, 7, "rand_sequence_packed.bin");

    free(data);
    return 0;
}