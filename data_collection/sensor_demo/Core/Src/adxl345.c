#include "adxl345.h"

// 使用 Cube 生成的 I2C1 句柄
extern I2C_HandleTypeDef hi2c1;

// 内部工具函数：写一个寄存器
static HAL_StatusTypeDef ADXL345_WriteReg(uint8_t reg, uint8_t val)
{
    return HAL_I2C_Mem_Write(&hi2c1,
                             ADXL345_I2C_ADDR,
                             reg,
                             I2C_MEMADD_SIZE_8BIT,
                             &val,
                             1,
                             100);
}

// 内部工具函数：读一个寄存器
static HAL_StatusTypeDef ADXL345_ReadReg(uint8_t reg, uint8_t *val)
{
    return HAL_I2C_Mem_Read(&hi2c1,
                            ADXL345_I2C_ADDR,
                            reg,
                            I2C_MEMADD_SIZE_8BIT,
                            val,
                            1,
                            100);
}

// 内部工具函数：连续读取多个字节
static HAL_StatusTypeDef ADXL345_ReadMulti(uint8_t startReg, uint8_t *buf, uint16_t len)
{
    return HAL_I2C_Mem_Read(&hi2c1,
                            ADXL345_I2C_ADDR,
                            startReg,
                            I2C_MEMADD_SIZE_8BIT,
                            buf,
                            len,
                            100);
}

// 传感器初始化
HAL_StatusTypeDef ADXL345_Init(void)
{
    HAL_StatusTypeDef ret;
    uint8_t id = 0;

    // 1. 读设备 ID 寄存器（0x00），正常应该是 0xE5
    ret = ADXL345_ReadReg(ADXL345_REG_DEVID, &id);
    if (ret != HAL_OK)
        return ret;

    // 你也可以在这里检查 id == 0xE5
    // if (id != 0xE5) return HAL_ERROR;

    // 2. 配置量程、分辨率：参考你给的代码 0x31 寄存器
    // 0x0B: FULL_RES=1（全分辨率），±16g，右对齐
    ret = ADXL345_WriteReg(ADXL345_REG_DATA_FORMAT, 0x0B);
    if (ret != HAL_OK)
        return ret;

    // 3. 设置输出速率：0x2C = 0x0B （大约 200 Hz，资料里有人写 100Hz，也没关系，后面可以调）
    ret = ADXL345_WriteReg(ADXL345_REG_BW_RATE, 0x0B);
    if (ret != HAL_OK)
        return ret;

    // 4. 使能测量：0x2D = 0x08（测量模式）
    ret = ADXL345_WriteReg(ADXL345_REG_POWER_CTL, 0x08);
    if (ret != HAL_OK)
        return ret;

    // 5. 也可以按照你那份代码写偏移寄存器（可选）
    //    OFSX(0x1E), OFSY(0x1F), OFSZ(0x20)
    //    这里先不写，有需要再加

    return HAL_OK;
}

// 读取原始 XYZ（单位：原始计数，13 位，补码）
HAL_StatusTypeDef ADXL345_ReadRaw(int16_t *x, int16_t *y, int16_t *z)
{
    uint8_t buf[6];
    HAL_StatusTypeDef ret;

    ret = ADXL345_ReadMulti(ADXL345_REG_DATAX0, buf, 6);
    if (ret != HAL_OK)
        return ret;

    // ADXL345 是 little endian：低字节在前
    *x = (int16_t)((buf[1] << 8) | buf[0]);
    *y = (int16_t)((buf[3] << 8) | buf[2]);
    *z = (int16_t)((buf[5] << 8) | buf[4]);

    return HAL_OK;
}

// 读取重力加速度（单位：g）
HAL_StatusTypeDef ADXL345_ReadG(float *x_g, float *y_g, float *z_g)
{
    int16_t x_raw, y_raw, z_raw;
    HAL_StatusTypeDef ret;

    ret = ADXL345_ReadRaw(&x_raw, &y_raw, &z_raw);
    if (ret != HAL_OK)
        return ret;

    // 在 FULL_RES 模式下，Scale 大约是 4mg/LSB = 0.004g
    const float scale = 0.004f;

    *x_g = x_raw * scale;
    *y_g = y_raw * scale;
    *z_g = z_raw * scale;

    return HAL_OK;
}
