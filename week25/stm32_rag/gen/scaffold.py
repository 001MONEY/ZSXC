"""从 vendor（STM32CubeF1）生成一个开箱即可编译的 Keil 工程。

为什么不用 CubeMX：本机没有 STM32CubeMX，而且课程要能离线复现。
vendor 里已经有完整的 CMSIS 头、启动文件、HAL 驱动和 `stm32f1xx_hal_conf.h` 模板，
把它们按需拷进交付目录，再生成一份 `.uvprojx`，就能直接用 `UV4 -b` 编译。

生成的目录（相对工程根）：

    Core/Inc      main.h / stm32f1xx_hal_conf.h / stm32f1xx_it.h
    Core/Src      main.c / stm32f1xx_it.c / system_stm32f1xx.c
    Core/Startup  startup_stm32f103xb.s
    Drivers       CMSIS + STM32F1xx_HAL_Driver（按用到的模块裁剪）
    MDK-ARM       <name>.uvprojx
"""

from __future__ import annotations

import shutil
from pathlib import Path
from string import Template

from ..core.config import HAL_DRIVER_DIR, HAL_ROOT
from .mcu import Device, resolve_device
from .project import KeilProject

CMSIS_INCLUDE = HAL_ROOT / "Drivers" / "CMSIS" / "Include"
CMSIS_DEVICE_INCLUDE = HAL_ROOT / "Drivers" / "CMSIS" / "Device" / "ST" / "STM32F1xx" / "Include"
CMSIS_STARTUP_DIR = HAL_ROOT / "Drivers" / "CMSIS" / "Device" / "ST" / "STM32F1xx" / "Source" / "Templates" / "arm"
CMSIS_SYSTEM_SRC = HAL_ROOT / "Drivers" / "CMSIS" / "Device" / "ST" / "STM32F1xx" / "Source" / "Templates" / "system_stm32f1xx.c"
HAL_SRC_DIR = HAL_DRIVER_DIR / "Src"
HAL_INC_DIR = HAL_DRIVER_DIR / "Inc"
HAL_CONF_TEMPLATE = (
    HAL_ROOT / "Projects" / "STM32F103RB-Nucleo" / "Examples" / "TIM" / "TIM_TimeBase" / "Inc" / "stm32f1xx_hal_conf.h"
)

# 任何工程都要的模块（HAL 内核、时钟、GPIO 基础设施）
HAL_BASE_SOURCES = (
    "stm32f1xx_hal.c",
    "stm32f1xx_hal_cortex.c",
    "stm32f1xx_hal_dma.c",
    "stm32f1xx_hal_flash.c",
    "stm32f1xx_hal_flash_ex.c",
    "stm32f1xx_hal_gpio.c",
    "stm32f1xx_hal_rcc.c",
    "stm32f1xx_hal_rcc_ex.c",
)

HAL_MODULE_SOURCES: dict[str, tuple[str, ...]] = {
    "tim": ("stm32f1xx_hal_tim.c", "stm32f1xx_hal_tim_ex.c"),
    "uart": ("stm32f1xx_hal_uart.c",),
    "usart": ("stm32f1xx_hal_usart.c",),
    "adc": ("stm32f1xx_hal_adc.c", "stm32f1xx_hal_adc_ex.c"),
    "i2c": ("stm32f1xx_hal_i2c.c",),
    "spi": ("stm32f1xx_hal_spi.c",),
    "iwdg": ("stm32f1xx_hal_iwdg.c",),
    "wwdg": ("stm32f1xx_hal_wwdg.c",),
    "rtc": ("stm32f1xx_hal_rtc.c", "stm32f1xx_hal_rtc_ex.c"),
    "pwr": ("stm32f1xx_hal_pwr.c",),
    "exti": ("stm32f1xx_hal_exti.c",),
}

# 模块 -> stm32f1xx_hal_conf.h 里要打开的开关
HAL_MODULE_CONF: dict[str, str] = {
    "tim": "HAL_TIM_MODULE_ENABLED",
    "uart": "HAL_UART_MODULE_ENABLED",
    "usart": "HAL_USART_MODULE_ENABLED",
    "adc": "HAL_ADC_MODULE_ENABLED",
    "i2c": "HAL_I2C_MODULE_ENABLED",
    "spi": "HAL_SPI_MODULE_ENABLED",
    "iwdg": "HAL_IWDG_MODULE_ENABLED",
    "wwdg": "HAL_WWDG_MODULE_ENABLED",
    "rtc": "HAL_RTC_MODULE_ENABLED",
    "pwr": "HAL_PWR_MODULE_ENABLED",
    "exti": "HAL_EXTI_MODULE_ENABLED",
}

UVPROJX_TEMPLATE = Template(
    """<?xml version="1.0" encoding="UTF-8" standalone="no" ?>
<Project xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="project_projx.xsd">

  <SchemaVersion>2.1</SchemaVersion>

  <Header>### uVision Project, (C) Keil Software</Header>

  <Targets>
    <Target>
      <TargetName>$target_name</TargetName>
      <ToolsetNumber>0x4</ToolsetNumber>
      <ToolsetName>ARM-ADS</ToolsetName>
      <pCCUsed>6160000::V6.16::ARMCLANG</pCCUsed>
      <uAC6>1</uAC6>
      <TargetOption>
        <TargetCommonOption>
          <Device>$device</Device>
          <Vendor>STMicroelectronics</Vendor>
          <PackID>Keil.STM32F1xx_DFP.2.4.1</PackID>
          <PackURL>https://www.keil.com/pack/</PackURL>
          <Cpu>IRAM(0x20000000,0x$ram_hex) IROM(0x08000000,0x$flash_hex) CPUTYPE("Cortex-M3") CLOCK($hse) ELITTLE</Cpu>
          <FlashUtilSpec></FlashUtilSpec>
          <StartupFile></StartupFile>
          <FlashDriverDll></FlashDriverDll>
          <DeviceId>0</DeviceId>
          <RegisterFile></RegisterFile>
          <MemoryEnv></MemoryEnv>
          <Cmp></Cmp>
          <Asm></Asm>
          <Linker></Linker>
          <OHString></OHString>
          <InfinionOptionDll></InfinionOptionDll>
          <SLE66CMisc></SLE66CMisc>
          <SLE66AMisc></SLE66AMisc>
          <SLE66LinkerMisc></SLE66LinkerMisc>
          <SFDFile></SFDFile>
          <bCustSvd>0</bCustSvd>
          <UseEnv>0</UseEnv>
          <BinPath></BinPath>
          <IncludePath></IncludePath>
          <LibPath></LibPath>
          <RegisterFilePath></RegisterFilePath>
          <DBRegisterFilePath></DBRegisterFilePath>
          <TargetStatus>
            <Error>0</Error>
            <ExitCodeStop>0</ExitCodeStop>
            <ButtonStop>0</ButtonStop>
            <NotGenerated>0</NotGenerated>
            <InvalidFlash>1</InvalidFlash>
          </TargetStatus>
          <OutputDirectory>.\\Objects\\</OutputDirectory>
          <OutputName>$output_name</OutputName>
          <CreateExecutable>1</CreateExecutable>
          <CreateLib>0</CreateLib>
          <CreateHexFile>1</CreateHexFile>
          <DebugInformation>1</DebugInformation>
          <BrowseInformation>0</BrowseInformation>
          <ListingPath>.\\Listings\\</ListingPath>
          <HexFormatSelection>1</HexFormatSelection>
          <Merge32K>0</Merge32K>
          <CreateBatchFile>0</CreateBatchFile>
          <BeforeCompile>
            <RunUserProg1>0</RunUserProg1>
            <RunUserProg2>0</RunUserProg2>
            <UserProg1Name></UserProg1Name>
            <UserProg2Name></UserProg2Name>
            <UserProg1Dos16Mode>0</UserProg1Dos16Mode>
            <UserProg2Dos16Mode>0</UserProg2Dos16Mode>
            <nStopU1X>0</nStopU1X>
            <nStopU2X>0</nStopU2X>
          </BeforeCompile>
          <BeforeMake>
            <RunUserProg1>0</RunUserProg1>
            <RunUserProg2>0</RunUserProg2>
            <UserProg1Name></UserProg1Name>
            <UserProg2Name></UserProg2Name>
            <UserProg1Dos16Mode>0</UserProg1Dos16Mode>
            <UserProg2Dos16Mode>0</UserProg2Dos16Mode>
            <nStopB1X>0</nStopB1X>
            <nStopB2X>0</nStopB2X>
          </BeforeMake>
          <AfterMake>
            <RunUserProg1>0</RunUserProg1>
            <RunUserProg2>0</RunUserProg2>
            <UserProg1Name></UserProg1Name>
            <UserProg2Name></UserProg2Name>
            <UserProg1Dos16Mode>0</UserProg1Dos16Mode>
            <UserProg2Dos16Mode>0</UserProg2Dos16Mode>
            <nStopA1X>0</nStopA1X>
            <nStopA2X>0</nStopA2X>
          </AfterMake>
          <SelectedForBatchBuild>0</SelectedForBatchBuild>
          <SVCSIdString></SVCSIdString>
        </TargetCommonOption>
        <CommonProperty>
          <UseCPPCompiler>0</UseCPPCompiler>
          <RVCTCodeConst>0</RVCTCodeConst>
          <RVCTZI>0</RVCTZI>
          <RVCTOtherData>0</RVCTOtherData>
          <ModuleSelection>0</ModuleSelection>
          <IncludeInBuild>1</IncludeInBuild>
          <AlwaysBuild>0</AlwaysBuild>
          <GenerateAssemblyFile>0</GenerateAssemblyFile>
          <AssembleAssemblyFile>0</AssembleAssemblyFile>
          <PublicsOnly>0</PublicsOnly>
          <StopOnExitCode>3</StopOnExitCode>
          <CustomArgument></CustomArgument>
          <IncludeLibraryModules></IncludeLibraryModules>
          <ComprImg>1</ComprImg>
        </CommonProperty>
        <DllOption>
          <SimDllName>SARMCM3.DLL</SimDllName>
          <SimDllArguments> -REMAP</SimDllArguments>
          <SimDlgDll>DCM.DLL</SimDlgDll>
          <SimDlgDllArguments>-pCM3</SimDlgDllArguments>
          <TargetDllName>SARMCM3.DLL</TargetDllName>
          <TargetDllArguments></TargetDllArguments>
          <TargetDlgDll>TCM.DLL</TargetDlgDll>
          <TargetDlgDllArguments>-pCM3</TargetDlgDllArguments>
        </DllOption>
        <DebugOption>
          <OPTHX>
            <HexSelection>1</HexSelection>
            <HexRangeLowAddress>0</HexRangeLowAddress>
            <HexRangeHighAddress>0</HexRangeHighAddress>
            <HexOffset>0</HexOffset>
            <Oh166RecLen>16</Oh166RecLen>
          </OPTHX>
        </DebugOption>
        <Utilities>
          <Flash1>
            <UseTargetDll>1</UseTargetDll>
            <UseExternalTool>0</UseExternalTool>
            <RunIndependent>0</RunIndependent>
            <UpdateFlashBeforeDebugging>1</UpdateFlashBeforeDebugging>
            <Capability>1</Capability>
            <DriverSelection>4096</DriverSelection>
          </Flash1>
          <bUseTDR>1</bUseTDR>
          <Flash2>BIN\\UL2CM3.DLL</Flash2>
          <Flash3>"" ()</Flash3>
          <Flash4></Flash4>
          <pFcarmOut></pFcarmOut>
          <pFcarmGrp></pFcarmGrp>
          <pFcArmRoot></pFcArmRoot>
          <FcArmLst>0</FcArmLst>
        </Utilities>
        <TargetArmAds>
          <ArmAdsMisc>
            <GenerateListings>0</GenerateListings>
            <asHll>1</asHll>
            <asAsm>1</asAsm>
            <asMacX>1</asMacX>
            <asSyms>1</asSyms>
            <asFals>1</asFals>
            <asDbgD>1</asDbgD>
            <asForm>1</asForm>
            <ldLst>0</ldLst>
            <ldmm>1</ldmm>
            <ldXref>1</ldXref>
            <BigEnd>0</BigEnd>
            <AdsALst>1</AdsALst>
            <AdsACrf>1</AdsACrf>
            <AdsANop>0</AdsANop>
            <AdsANot>0</AdsANot>
            <AdsLLst>1</AdsLLst>
            <AdsLmap>1</AdsLmap>
            <AdsLcgr>1</AdsLcgr>
            <AdsLsym>1</AdsLsym>
            <AdsLszi>1</AdsLszi>
            <AdsLtoi>1</AdsLtoi>
            <AdsLsun>1</AdsLsun>
            <AdsLven>1</AdsLven>
            <AdsLsxf>1</AdsLsxf>
            <RvctClst>0</RvctClst>
            <GenPPlst>0</GenPPlst>
            <AdsCpuType>"Cortex-M3"</AdsCpuType>
            <RvctDeviceName></RvctDeviceName>
            <mOS>0</mOS>
            <uocRom>0</uocRom>
            <uocRam>0</uocRam>
            <hadIROM>1</hadIROM>
            <hadIRAM>1</hadIRAM>
            <hadXRAM>0</hadXRAM>
            <uocXRam>0</uocXRam>
            <RvdsVP>0</RvdsVP>
            <RvdsMve>0</RvdsMve>
            <RvdsCdeCp>0</RvdsCdeCp>
            <hadIRAM2>0</hadIRAM2>
            <hadIROM2>0</hadIROM2>
            <StupSel>8</StupSel>
            <useUlib>1</useUlib>
            <EndSel>0</EndSel>
            <uLtcg>0</uLtcg>
            <nSecure>0</nSecure>
            <RoSelD>3</RoSelD>
            <RwSelD>3</RwSelD>
            <CodeSel>0</CodeSel>
            <OptFeed>0</OptFeed>
            <NoZi1>0</NoZi1>
            <NoZi2>0</NoZi2>
            <NoZi3>0</NoZi3>
            <NoZi4>0</NoZi4>
            <NoZi5>0</NoZi5>
            <Ro1Chk>0</Ro1Chk>
            <Ro2Chk>0</Ro2Chk>
            <Ro3Chk>0</Ro3Chk>
            <Ir1Chk>1</Ir1Chk>
            <Ir2Chk>0</Ir2Chk>
            <Ra1Chk>0</Ra1Chk>
            <Ra2Chk>0</Ra2Chk>
            <Ra3Chk>0</Ra3Chk>
            <Im1Chk>1</Im1Chk>
            <Im2Chk>0</Im2Chk>
            <OnChipMemories>
              <Ocm1>
                <Type>0</Type>
                <StartAddress>0x0</StartAddress>
                <Size>0x0</Size>
              </Ocm1>
              <Ocm2>
                <Type>0</Type>
                <StartAddress>0x0</StartAddress>
                <Size>0x0</Size>
              </Ocm2>
              <Ocm3>
                <Type>0</Type>
                <StartAddress>0x0</StartAddress>
                <Size>0x0</Size>
              </Ocm3>
              <Ocm4>
                <Type>0</Type>
                <StartAddress>0x0</StartAddress>
                <Size>0x0</Size>
              </Ocm4>
              <Ocm5>
                <Type>0</Type>
                <StartAddress>0x0</StartAddress>
                <Size>0x0</Size>
              </Ocm5>
              <Ocm6>
                <Type>0</Type>
                <StartAddress>0x0</StartAddress>
                <Size>0x0</Size>
              </Ocm6>
              <IRAM>
                <Type>0</Type>
                <StartAddress>0x20000000</StartAddress>
                <Size>0x$ram_hex</Size>
              </IRAM>
              <IROM>
                <Type>1</Type>
                <StartAddress>0x8000000</StartAddress>
                <Size>0x$flash_hex</Size>
              </IROM>
              <XRAM>
                <Type>0</Type>
                <StartAddress>0x0</StartAddress>
                <Size>0x0</Size>
              </XRAM>
              <OCR_RVCT1>
                <Type>1</Type>
                <StartAddress>0x0</StartAddress>
                <Size>0x0</Size>
              </OCR_RVCT1>
              <OCR_RVCT2>
                <Type>1</Type>
                <StartAddress>0x0</StartAddress>
                <Size>0x0</Size>
              </OCR_RVCT2>
              <OCR_RVCT3>
                <Type>1</Type>
                <StartAddress>0x0</StartAddress>
                <Size>0x0</Size>
              </OCR_RVCT3>
              <OCR_RVCT4>
                <Type>1</Type>
                <StartAddress>0x8000000</StartAddress>
                <Size>0x$flash_hex</Size>
              </OCR_RVCT4>
              <OCR_RVCT5>
                <Type>1</Type>
                <StartAddress>0x0</StartAddress>
                <Size>0x0</Size>
              </OCR_RVCT5>
              <OCR_RVCT6>
                <Type>0</Type>
                <StartAddress>0x0</StartAddress>
                <Size>0x0</Size>
              </OCR_RVCT6>
              <OCR_RVCT7>
                <Type>0</Type>
                <StartAddress>0x0</StartAddress>
                <Size>0x0</Size>
              </OCR_RVCT7>
              <OCR_RVCT8>
                <Type>0</Type>
                <StartAddress>0x0</StartAddress>
                <Size>0x0</Size>
              </OCR_RVCT8>
              <OCR_RVCT9>
                <Type>0</Type>
                <StartAddress>0x20000000</StartAddress>
                <Size>0x$ram_hex</Size>
              </OCR_RVCT9>
              <OCR_RVCT10>
                <Type>0</Type>
                <StartAddress>0x0</StartAddress>
                <Size>0x0</Size>
              </OCR_RVCT10>
            </OnChipMemories>
            <RvctStartVector></RvctStartVector>
          </ArmAdsMisc>
          <Cads>
            <interw>1</interw>
            <Optim>1</Optim>
            <oTime>0</oTime>
            <SplitLS>0</SplitLS>
            <OneElfS>1</OneElfS>
            <Strict>0</Strict>
            <EnumInt>0</EnumInt>
            <PlainCh>0</PlainCh>
            <Ropi>0</Ropi>
            <Rwpi>0</Rwpi>
            <wLevel>3</wLevel>
            <uThumb>0</uThumb>
            <uSurpInc>0</uSurpInc>
            <uC99>1</uC99>
            <uGnu>0</uGnu>
            <useXO>0</useXO>
            <v6Lang>3</v6Lang>
            <v6LangP>3</v6LangP>
            <vShortEn>1</vShortEn>
            <vShortWch>1</vShortWch>
            <v6Lto>0</v6Lto>
            <v6WtE>0</v6WtE>
            <v6Rtti>0</v6Rtti>
            <VariousControls>
              <MiscControls></MiscControls>
              <Define>$defines</Define>
              <Undefine></Undefine>
              <IncludePath>$includes</IncludePath>
            </VariousControls>
          </Cads>
          <Aads>
            <interw>1</interw>
            <Ropi>0</Ropi>
            <Rwpi>0</Rwpi>
            <thumb>0</thumb>
            <SplitLS>0</SplitLS>
            <SwStkChk>0</SwStkChk>
            <NoWarn>0</NoWarn>
            <uSurpInc>0</uSurpInc>
            <useXO>0</useXO>
            <ClangAsOpt>4</ClangAsOpt>
            <VariousControls>
              <MiscControls></MiscControls>
              <Define></Define>
              <Undefine></Undefine>
              <IncludePath></IncludePath>
            </VariousControls>
          </Aads>
          <LDads>
            <umfTarg>1</umfTarg>
            <Ropi>0</Ropi>
            <Rwpi>0</Rwpi>
            <noStLib>0</noStLib>
            <RepFail>1</RepFail>
            <useFile>0</useFile>
            <TextAddressRange>0x08000000</TextAddressRange>
            <DataAddressRange>0x20000000</DataAddressRange>
            <pXoBase></pXoBase>
            <ScatterFile></ScatterFile>
            <IncludeLibs></IncludeLibs>
            <IncludeLibsPath></IncludeLibsPath>
            <Misc></Misc>
            <LinkerInputFile></LinkerInputFile>
            <DisabledWarnings></DisabledWarnings>
          </LDads>
        </TargetArmAds>
      </TargetOption>
      <Groups>
$groups
      </Groups>
    </Target>
  </Targets>

</Project>
"""
)

MAIN_H_TEMPLATE = """/* main.h —— 由 stm32_rag 代码生成器生成 */
#ifndef __MAIN_H
#define __MAIN_H

#ifdef __cplusplus
extern "C" {
#endif

#include "stm32f1xx_hal.h"

void Error_Handler(void);

#ifdef __cplusplus
}
#endif

#endif /* __MAIN_H */
"""

MAIN_C_TEMPLATE = """/* main.c —— 由 stm32_rag 代码生成器生成，可在 USER CODE 段内继续添加逻辑 */
#include "main.h"

static void SystemClock_Config(void);

int main(void)
{
  HAL_Init();
  SystemClock_Config();

  while (1)
  {
  }
}

/* 72 MHz：HSE 8 MHz → PLL ×9 */
static void SystemClock_Config(void)
{
  RCC_OscInitTypeDef RCC_OscInitStruct = {0};
  RCC_ClkInitTypeDef RCC_ClkInitStruct = {0};

  RCC_OscInitStruct.OscillatorType = RCC_OSCILLATORTYPE_HSE;
  RCC_OscInitStruct.HSEState = RCC_HSE_ON;
  RCC_OscInitStruct.HSEPredivValue = RCC_HSE_PREDIV_DIV1;
  RCC_OscInitStruct.HSIState = RCC_HSI_ON;
  RCC_OscInitStruct.PLL.PLLState = RCC_PLL_ON;
  RCC_OscInitStruct.PLL.PLLSource = RCC_PLLSOURCE_HSE;
  RCC_OscInitStruct.PLL.PLLMUL = RCC_PLL_MUL9;
  if (HAL_RCC_OscConfig(&RCC_OscInitStruct) != HAL_OK)
  {
    Error_Handler();
  }

  RCC_ClkInitStruct.ClockType = RCC_CLOCKTYPE_HCLK | RCC_CLOCKTYPE_SYSCLK
                              | RCC_CLOCKTYPE_PCLK1 | RCC_CLOCKTYPE_PCLK2;
  RCC_ClkInitStruct.SYSCLKSource = RCC_SYSCLKSOURCE_PLLCLK;
  RCC_ClkInitStruct.AHBCLKDivider = RCC_SYSCLK_DIV1;
  RCC_ClkInitStruct.APB1CLKDivider = RCC_HCLK_DIV2;
  RCC_ClkInitStruct.APB2CLKDivider = RCC_HCLK_DIV1;
  if (HAL_RCC_ClockConfig(&RCC_ClkInitStruct, FLASH_LATENCY_2) != HAL_OK)
  {
    Error_Handler();
  }
}

void Error_Handler(void)
{
  __disable_irq();
  while (1)
  {
  }
}
"""

IT_H_TEMPLATE = """/* stm32f1xx_it.h —— 中断服务函数声明 */
#ifndef __STM32F1xx_IT_H
#define __STM32F1xx_IT_H

#ifdef __cplusplus
extern "C" {
#endif

void NMI_Handler(void);
void HardFault_Handler(void);
void MemManage_Handler(void);
void BusFault_Handler(void);
void UsageFault_Handler(void);
void SVC_Handler(void);
void DebugMon_Handler(void);
void PendSV_Handler(void);
void SysTick_Handler(void);

#ifdef __cplusplus
}
#endif

#endif /* __STM32F1xx_IT_H */
"""

IT_C_TEMPLATE = """/* stm32f1xx_it.c —— 默认只保留 SysTick，外设中断由代码生成器追加 */
#include "main.h"
#include "stm32f1xx_it.h"

void NMI_Handler(void)
{
}

void HardFault_Handler(void)
{
  while (1)
  {
  }
}

void MemManage_Handler(void)
{
  while (1)
  {
  }
}

void BusFault_Handler(void)
{
  while (1)
  {
  }
}

void UsageFault_Handler(void)
{
  while (1)
  {
  }
}

void SVC_Handler(void)
{
}

void DebugMon_Handler(void)
{
}

void PendSV_Handler(void)
{
}

void SysTick_Handler(void)
{
  HAL_IncTick();
}
"""


def _copy_tree(src: Path, dst: Path) -> int:
    """递归复制目录内容（HAL 的 Inc 下有 Legacy 子目录，必须一并带上）。"""
    if not src.is_dir():
        raise FileNotFoundError(f"vendor 缺少目录：{src}")
    dst.mkdir(parents=True, exist_ok=True)
    count = 0
    for item in src.rglob("*"):
        if not item.is_file():
            continue
        target = dst / item.relative_to(src)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(item, target)
        count += 1
    return count


def scaffold_project(
    dest: str | Path,
    device: str | Device | None = None,
    name: str = "stm32_app",
    modules: tuple[str, ...] = ("tim",),
    hse: int = 8_000_000,
) -> dict:
    """在 `dest` 生成一个可直接编译的 Keil 工程，返回关键路径。"""
    dest = Path(dest).expanduser().resolve()
    dev = device if isinstance(device, Device) else resolve_device(device)

    core_inc = dest / "Core" / "Inc"
    core_src = dest / "Core" / "Src"
    startup_dir = dest / "Core" / "Startup"
    hal_src = dest / "Drivers" / "STM32F1xx_HAL_Driver" / "Src"
    hal_inc = dest / "Drivers" / "STM32F1xx_HAL_Driver" / "Inc"
    cmsis_inc = dest / "Drivers" / "CMSIS" / "Include"
    cmsis_dev = dest / "Drivers" / "CMSIS" / "Device" / "ST" / "STM32F1xx" / "Include"
    mdk = dest / "MDK-ARM"

    for path in (core_inc, core_src, startup_dir, hal_src, hal_inc, cmsis_inc, cmsis_dev, mdk):
        path.mkdir(parents=True, exist_ok=True)

    # CMSIS / 启动文件 / system
    _copy_tree(CMSIS_INCLUDE, cmsis_inc)
    _copy_tree(CMSIS_DEVICE_INCLUDE, cmsis_dev)
    startup_src = CMSIS_STARTUP_DIR / dev.startup
    if not startup_src.is_file():
        raise FileNotFoundError(f"vendor 里没有启动文件：{startup_src}")
    shutil.copy2(startup_src, startup_dir / dev.startup)
    shutil.copy2(CMSIS_SYSTEM_SRC, core_src / "system_stm32f1xx.c")

    # HAL：头文件全带，源文件按模块裁剪
    _copy_tree(HAL_INC_DIR, hal_inc)
    for source in HAL_BASE_SOURCES:
        shutil.copy2(HAL_SRC_DIR / source, hal_src / source)
    added = ensure_hal_modules(dest, modules)
    enable_hal_conf(core_inc / "stm32f1xx_hal_conf.h", modules)

    # 初始源码（随后会被代码生成器覆盖）
    (core_inc / "main.h").write_text(MAIN_H_TEMPLATE, encoding="utf-8")
    (core_inc / "stm32f1xx_it.h").write_text(IT_H_TEMPLATE, encoding="utf-8")
    (core_src / "main.c").write_text(MAIN_C_TEMPLATE, encoding="utf-8")
    (core_src / "stm32f1xx_it.c").write_text(IT_C_TEMPLATE, encoding="utf-8")

    project = _write_uvprojx(dest, dev, name=name, modules=modules, hse=hse)
    return {
        "workspace": str(dest),
        "project": str(project.path),
        "device": dev.name,
        "hal_sources": added,
        "files": [str(path.relative_to(dest)) for path in dest.rglob("*") if path.is_file()],
    }


def ensure_hal_modules(dest: str | Path, modules: tuple[str, ...] | list[str]) -> list[str]:
    """把模块对应的 HAL 源文件补进 Drivers 目录，返回本次新增的文件名。"""
    dest = Path(dest).expanduser().resolve()
    hal_src = dest / "Drivers" / "STM32F1xx_HAL_Driver" / "Src"
    hal_src.mkdir(parents=True, exist_ok=True)
    added: list[str] = []
    for module in modules:
        for source in HAL_MODULE_SOURCES.get(module, ()):
            target = hal_src / source
            if target.is_file():
                continue
            origin = HAL_SRC_DIR / source
            if not origin.is_file():
                raise FileNotFoundError(f"vendor 缺少 HAL 源文件：{origin}")
            shutil.copy2(origin, target)
            added.append(source)
    return added


def enable_hal_conf(conf_path: str | Path, modules: tuple[str, ...] | list[str]) -> list[str]:
    """打开 `stm32f1xx_hal_conf.h` 里对应模块的开关，返回本次打开的宏。"""
    import re

    conf_path = Path(conf_path)
    if not conf_path.is_file():
        if HAL_CONF_TEMPLATE.is_file():
            shutil.copy2(HAL_CONF_TEMPLATE, conf_path)
        else:
            raise FileNotFoundError(f"找不到 HAL 配置模板：{HAL_CONF_TEMPLATE}")
    text = conf_path.read_text(encoding="utf-8")
    enabled: list[str] = []
    for module in modules:
        macro = HAL_MODULE_CONF.get(module)
        if not macro:
            continue
        # CubeMX 各版本的注释写法不同：`/* #define X */`、`/*#define X   */` 都有
        commented = re.compile(rf"/\*\s*#define\s+{macro}\b[^*]*\*/")
        active = re.compile(rf"(?m)^\s*#define\s+{macro}\b")
        if active.search(text):
            continue
        if commented.search(text):
            text = commented.sub(f"#define {macro}", text, count=1)
            enabled.append(macro)
            continue
        marker = "#define HAL_MODULE_ENABLED\n"
        if marker in text:
            text = text.replace(marker, f"{marker}#define {macro}\n", 1)
            enabled.append(macro)
    conf_path.write_text(text, encoding="utf-8")
    return enabled


def _write_uvprojx(dest: Path, device: Device, name: str, modules: tuple[str, ...], hse: int) -> KeilProject:
    dest = Path(dest).expanduser().resolve()
    mdk = dest / "MDK-ARM"
    mdk.mkdir(parents=True, exist_ok=True)
    path = mdk / f"{name}.uvprojx"

    hal_sources = list(HAL_BASE_SOURCES)
    for module in modules:
        hal_sources.extend(HAL_MODULE_SOURCES.get(module, ()))
    hal_files = [_file_block("Drivers/STM32F1xx_HAL_Driver/Src", src, 1) for src in dict.fromkeys(hal_sources)]

    groups = "\n".join(
        [
            _group_block("Startup", [_file_block("Core/Startup", device.startup, 2)]),
            _group_block("CMSIS", [_file_block("Core/Src", "system_stm32f1xx.c", 1)]),
            _group_block(
                "Application/User",
                [_file_block("Core/Src", "main.c", 1), _file_block("Core/Src", "stm32f1xx_it.c", 1)],
            ),
            _group_block("Drivers/STM32F1xx_HAL_Driver", hal_files),
        ]
    )

    includes = ";".join(
        [
            r"..\Core\Inc",
            r"..\Drivers\CMSIS\Include",
            r"..\Drivers\CMSIS\Device\ST\STM32F1xx\Include",
            r"..\Drivers\STM32F1xx_HAL_Driver\Inc",
        ]
    )
    text = UVPROJX_TEMPLATE.substitute(
        target_name=name,
        device=device.name,
        ram_hex=f"{device.ram_bytes:08X}",
        flash_hex=f"{device.flash_bytes:08X}",
        hse=hse,
        output_name=name,
        defines=f"USE_HAL_DRIVER,{device.define}",
        includes=includes,
        groups=groups,
    )
    path.write_bytes(text.encode("utf-8"))  # 无 BOM，UV4 才认
    return KeilProject(path)


def _file_block(folder: str, name: str, file_type: int) -> str:
    return (
        "            <File>\n"
        f"              <FileName>{name}</FileName>\n"
        f"              <FileType>{file_type}</FileType>\n"
        f"              <FilePath>..\\{folder.replace('/', chr(92))}\\{name}</FilePath>\n"
        "            </File>"
    )


def _group_block(name: str, files: list[str]) -> str:
    body = "\n".join(files)
    return (
        "        <Group>\n"
        f"          <GroupName>{name}</GroupName>\n"
        "          <Files>\n"
        f"{body}\n"
        "          </Files>\n"
        "        </Group>"
    )
