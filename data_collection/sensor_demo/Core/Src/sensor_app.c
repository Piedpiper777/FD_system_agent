#include "sensor_app.h"
#include "adxl345.h"
#include <stdio.h>

extern UART_HandleTypeDef huart1;
extern I2C_HandleTypeDef hi2c1;

void Sensor_App_Init(void)
{
    HAL_StatusTypeDef ret;
    uint8_t dev_id = 0;
    
    // 初始化加速度计
    ret = ADXL345_Init();
    
    // 发送初始化状态
    char init_buf[128];
    int init_len = snprintf(init_buf, sizeof(init_buf), 
                           "ADXL345 Init: %s\r\n", 
                           (ret == HAL_OK) ? "OK" : "FAIL");
    HAL_UART_Transmit(&huart1, (uint8_t*)init_buf, init_len, 1000);
    
    // 尝试读取设备ID验证连接
    if (HAL_I2C_Mem_Read(&hi2c1, (0x53 << 1), 0x00, I2C_MEMADD_SIZE_8BIT, &dev_id, 1, 100) == HAL_OK)
    {
        int id_len = snprintf(init_buf, sizeof(init_buf), "Device ID: 0x%02X (expected 0xE5)\r\n", dev_id);
        HAL_UART_Transmit(&huart1, (uint8_t*)init_buf, id_len, 1000);
    }
    
    // 等待传感器稳定
    HAL_Delay(100);
}

void Sensor_App_Loop(void)
{
    uint32_t ts = HAL_GetTick();   // 毫秒时间戳

    // 读取传感器原始数据
    uint8_t data_buf[6];
    HAL_StatusTypeDef ret = HAL_I2C_Mem_Read(&hi2c1, (0x53 << 1), 0x32, 
                                             I2C_MEMADD_SIZE_8BIT, 
                                             data_buf, 6, 100);
    
    if (ret != HAL_OK)
    {
        // 读取失败时，发送错误信息以便调试
        char err_buf[64];
        int err_len = snprintf(err_buf, sizeof(err_buf), 
                              "%lu,ERROR,ERROR,ERROR\r\n", 
                              (unsigned long)ts);
        HAL_UART_Transmit(&huart1, (uint8_t*)err_buf, err_len, 100);
        HAL_Delay(100);
        return;
    }
    
    // 解析原始数据（ADXL345是小端序：低字节在前）
    int16_t x_raw = (int16_t)((data_buf[1] << 8) | data_buf[0]);
    int16_t y_raw = (int16_t)((data_buf[3] << 8) | data_buf[2]);
    int16_t z_raw = (int16_t)((data_buf[5] << 8) | data_buf[4]);
    
    // 转换为g值（使用整数运算避免浮点格式化问题）
    // scale = 0.004 = 4/1000，所以 g = raw * 4 / 1000
    int32_t ax_mg = (int32_t)x_raw * 4;  // 单位：mg (millig)
    int32_t ay_mg = (int32_t)y_raw * 4;
    int32_t az_mg = (int32_t)z_raw * 4;
    
    // 使用整数格式化，避免浮点数问题
    int ax_int = ax_mg / 1000;
    int ax_frac = (ax_mg < 0 ? -ax_mg : ax_mg) % 1000;
    int ay_int = ay_mg / 1000;
    int ay_frac = (ay_mg < 0 ? -ay_mg : ay_mg) % 1000;
    int az_int = az_mg / 1000;
    int az_frac = (az_mg < 0 ? -az_mg : az_mg) % 1000;
    
    // 格式化输出（只输出加速度数据，不包含温度）
    char buf[128];
    int len = snprintf(
        buf, sizeof(buf),
        "%lu,%d.%03d,%d.%03d,%d.%03d\r\n",
        (unsigned long)ts, ax_int, ax_frac, ay_int, ay_frac, az_int, az_frac
    );

    HAL_UART_Transmit(&huart1, (uint8_t*)buf, len, 100);
    HAL_Delay(100); // 10Hz 输出
}
