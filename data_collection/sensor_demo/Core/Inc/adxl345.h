#pragma once

#include "main.h"

// ADXL345 7 位地址是 0x53，
// HAL 里用的是 8 位地址（左移一位），所以：
#define ADXL345_I2C_ADDR   (0x53 << 1)

// 一些常用寄存器地址
#define ADXL345_REG_DEVID       0x00
#define ADXL345_REG_DATA_FORMAT 0x31
#define ADXL345_REG_BW_RATE     0x2C
#define ADXL345_REG_POWER_CTL   0x2D
#define ADXL345_REG_DATAX0      0x32

// 对外暴露的 API（给 sensor_app.c 用）
HAL_StatusTypeDef ADXL345_Init(void);
HAL_StatusTypeDef ADXL345_ReadRaw(int16_t *x, int16_t *y, int16_t *z);
HAL_StatusTypeDef ADXL345_ReadG(float *x_g, float *y_g, float *z_g);
