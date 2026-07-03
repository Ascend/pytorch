# Usage Instructions

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-06-12T08:25:16.192Z pushedAt=2026-06-12T11:22:41.063Z -->

## Generation Mechanism

When users encounter errors during scenarios such as model building, inference execution, or training scripts, for example, detecting input errors (command-line input parameter errors, API input parameter errors, input file errors, unsupported operators, unsupported Shape/Format, etc.) or environment errors, the system prints the error code on the user interface. In actual problem locating, it is necessary to combine the specific error message and plog information for diagnosis.

## Error Code Format

> [!NOTE]
>
> Generally, ERR\*\*005 indicates an internal error. For internal errors, you can contact Huawei for troubleshooting. You can submit an issue in the [Ascend Community](https://gitcode.com/Ascend/pytorch/issues) for assistance.
> The error code-related information described in this document is fully displayed when the error is printed on the screen, including possible causes and solutions. Therefore, this section only lists the relevant content for reference.

Due to different scenarios, use cases, and causes of failures, the printed error code information varies. Therefore, this document uses the \[%s\] variable format to replace the actual print logs. The specific logs shall be subject to the screen print.

For example, the representation of the **ERR00002** error code in the manual is:

\[ERROR\] \[%s\] \(PID:\[%s\], Device:\[%s\], RankID:\[%s\]\) ERR\[%s\]\[%s\] \[%s\] \[%s\]

An example of the actual error message on the user interface is shown below. The error message consists of five parts. For details, see [Table 1](#error-message-detailed-description).

```ColdFusion
[ERROR] 2024-03-07-01:31:48 (PID:116072, Device:0, RankID:-1) ERR00002 PTA invalid type
```

**Table 1**  Error message details<a id="error-message-detailed-description"></a>

|Error Message|Description|
|:---|:---|
|Log level|Example: [ERROR].|
|Log timestamp|Example: 2024-03-07-01:31:48.|
|Device information|Example: (PID:116072, Device:0, RankID:-1).<br>- PID: Process ID of the error process.<br>- Device: Device number where the error process resides, obtained through the ACL interface. If acquisition fails, the default value -1 is printed.<br>- RankID: Rank of the device where the error process resides within the communication domain, obtained through the environment variable RANK. If not configured, the default value -1 is printed.|
|Error code|Example: ERR00002.<br>The error code is represented as an 8-character string. For the field description, see [Figure 1](#field-description):<br>- Field 1 is ERR, which indicates an error class.<br>- Field 2 is a two-digit number indicating the error module. For details, see [Figure 2](#field-2-meaning).<br>- Field 3 is a three-digit number indicating the error type. For details, see [Figure 3](#field-3-meaning).|
|Error code description|Example: PTA invalid type.|

**Figure 1**  Field description<a id="field-description"></a>  
![figure 1](../figures/description_each_field.png)

**Figure 2**  Field 2 meaning<a id="field-2-meaning"></a>  
![figure 2](../figures/meaning_field_2.png)

**Figure 3**  Field 3 meaning<a id="field-3-meaning"></a>  
![figure 3](../figures/meaning_field_3.png)
