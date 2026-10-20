# MiniFact 逐句测试集

原文及标注来自固定的 `cases.json`。下列区间均为 `[start_idx, end_idx)`，重复位置完整保留；参考事实数按不同命题计数。

## 基础复合句

检查并列事实拆分，以及时间、数量是否保留。

### base_01

> On Monday, Aurora City opened a new library, and the building has 200 seats.

参考事实：2 条；限定：2 项。

**事实 1　`library`**

- 参考引用：`On Monday, Aurora City opened a new library`
- 全部位置：[0, 43)（1 处）
- 核心证据：`Aurora City opened a new library`
- 限定：时间 `On Monday`

**事实 2　`seats`**

- 参考引用：`the building has 200 seats`
- 全部位置：[49, 75)（1 处）
- 核心证据：`the building has`、`seats`
- 限定：数量 `200`
- 上下文：the building 指前文新图书馆。

### base_02

> Riverdale Hospital added 40 beds in June, but its emergency department stayed closed.

参考事实：2 条；限定：2 项。

**事实 1　`beds`**

- 参考引用：`Riverdale Hospital added 40 beds in June`
- 全部位置：[0, 40)（1 处）
- 核心证据：`Riverdale Hospital added`、`beds`
- 限定：数量 `40`；时间 `in June`

**事实 2　`emergency`**

- 参考引用：`its emergency department stayed closed`
- 全部位置：[46, 84)（1 处）
- 核心证据：`its emergency department stayed closed`
- 限定：无单独标注的限定
- 上下文：its 指 Riverdale Hospital。

### base_03

> Green Line ridership rose 12% in 2025, while ticket prices fell by 5%.

参考事实：2 条；限定：3 项。

**事实 1　`ridership`**

- 参考引用：`Green Line ridership rose 12% in 2025`
- 全部位置：[0, 37)（1 处）
- 核心证据：`Green Line ridership rose`
- 限定：数量 `12%`；时间 `in 2025`

**事实 2　`prices`**

- 参考引用：`ticket prices fell by 5%`
- 全部位置：[45, 69)（1 处）
- 核心证据：`ticket prices fell`
- 限定：数量 `by 5%`
- 上下文：票价所属线路依赖整句语境；不自行给第二分句补写年份。

## 重复引文

同一事实重复出现只计一个命题；引用存在多个位置时不得猜测。

### repeat_01

> North Hall has 80 seats. North Hall has 80 seats.

参考事实：1 条；限定：1 项。

**事实 1　`hall`**

- 参考引用：`North Hall has 80 seats`
- 全部位置：[0, 23)、[25, 48)（2 处）
- 核心证据：`North Hall has`、`seats`
- 限定：数量 `80`
- 上下文：同一命题有两处原文提及，事实分母为 1；两处位置均保留。

### repeat_02

> "The clinic stayed open in June"; "The clinic stayed open in June"; its pharmacy stayed closed.

参考事实：2 条；限定：1 项。

**事实 1　`clinic`**

- 参考引用：`The clinic stayed open in June`
- 全部位置：[1, 31)、[35, 65)（2 处）
- 核心证据：`The clinic stayed open`
- 限定：时间 `in June`
- 上下文：两段相同引文表达同一命题，不能按出现次数重复计数。

**事实 2　`pharmacy`**

- 参考引用：`its pharmacy stayed closed`
- 全部位置：[68, 94)（1 处）
- 核心证据：`its pharmacy stayed closed`
- 限定：无单独标注的限定
- 上下文：its 指前文诊所。

## 共享主语

第二个谓语沿用前文主语；检查是否擅自补写主语而破坏逐字引用。

### shared_01

> Meadow Clinic hired 12 nurses in March and opened a new ward in July.

参考事实：2 条；限定：3 项。

**事实 1　`nurses`**

- 参考引用：`Meadow Clinic hired 12 nurses in March`
- 全部位置：[0, 38)（1 处）
- 核心证据：`Meadow Clinic hired`、`nurses`
- 限定：数量 `12`；时间 `in March`

**事实 2　`ward`**

- 参考引用：`opened a new ward in July`
- 全部位置：[43, 68)（1 处）
- 核心证据：`opened a new ward`
- 限定：时间 `in July`
- 上下文：沿用 Meadow Clinic 作为主语；不能在原文引用中补写该主语。

### shared_02

> Eastport Council approved the park on Tuesday but did not approve the road on Wednesday.

参考事实：2 条；限定：3 项。

**事实 1　`park`**

- 参考引用：`Eastport Council approved the park on Tuesday`
- 全部位置：[0, 45)（1 处）
- 核心证据：`Eastport Council approved the park`
- 限定：时间 `on Tuesday`

**事实 2　`road`**

- 参考引用：`did not approve the road on Wednesday`
- 全部位置：[50, 87)（1 处）
- 核心证据：`approve the road`
- 限定：否定 `did not`；时间 `on Wednesday`
- 上下文：沿用 Eastport Council 作为主语；同时保留 did not。

## 指代

保留 she、its、that wing 等原文表达，并注明其上下文指向。

### pronoun_01

> Lina opened a bakery in April, and she hired two bakers in May.

参考事实：2 条；限定：3 项。

**事实 1　`bakery`**

- 参考引用：`Lina opened a bakery in April`
- 全部位置：[0, 29)（1 处）
- 核心证据：`Lina opened a bakery`
- 限定：时间 `in April`

**事实 2　`bakers`**

- 参考引用：`she hired two bakers in May`
- 全部位置：[35, 62)（1 处）
- 核心证据：`she hired`、`bakers`
- 限定：数量 `two`；时间 `in May`
- 上下文：she 指 Lina。

### pronoun_02

> Harbor Hospital closed its west wing in July, but that wing reopened in October.

参考事实：2 条；限定：2 项。

**事实 1　`closure`**

- 参考引用：`Harbor Hospital closed its west wing in July`
- 全部位置：[0, 44)（1 处）
- 核心证据：`Harbor Hospital closed its west wing`
- 限定：时间 `in July`

**事实 2　`reopening`**

- 参考引用：`that wing reopened in October`
- 全部位置：[50, 79)（1 处）
- 核心证据：`that wing reopened`
- 限定：时间 `in October`
- 上下文：that wing 指前文 west wing。

## 否定

区分事实核心覆盖与否定条件保留；检查 did not 和 No。

### negation_01

> The board did not cancel the concert, and tickets remained valid until Friday.

参考事实：2 条；限定：2 项。

**事实 1　`concert`**

- 参考引用：`The board did not cancel the concert`
- 全部位置：[0, 36)（1 处）
- 核心证据：`The board`、`cancel the concert`
- 限定：否定 `did not`

**事实 2　`tickets`**

- 参考引用：`tickets remained valid until Friday`
- 全部位置：[42, 77)（1 处）
- 核心证据：`tickets remained valid`
- 限定：时间 `until Friday`

### negation_02

> No trains ran on Sunday, but buses carried 300 passengers.

参考事实：2 条；限定：3 项。

**事实 1　`trains`**

- 参考引用：`No trains ran on Sunday`
- 全部位置：[0, 23)（1 处）
- 核心证据：`trains`、`ran`
- 限定：否定 `No`；时间 `on Sunday`

**事实 2　`buses`**

- 参考引用：`buses carried 300 passengers`
- 全部位置：[29, 57)（1 处）
- 核心证据：`buses carried`、`passengers`
- 限定：数量 `300`

## 复杂时间

覆盖时间区间、钟点、相对日期、未来模态和适用范围。

### time_01

> From March 3 to March 8, the museum closed at 4 p.m., while its cafe stayed open until 6 p.m. during that period.

参考事实：2 条；限定：4 项。

**事实 1　`museum`**

- 参考引用：`From March 3 to March 8, the museum closed at 4 p.m.`
- 全部位置：[0, 52)（1 处）
- 核心证据：`the museum closed`
- 限定：时间 `From March 3 to March 8`；时间 `at 4 p.m.`

**事实 2　`cafe`**

- 参考引用：`its cafe stayed open until 6 p.m. during that period`
- 全部位置：[60, 112)（1 处）
- 核心证据：`its cafe stayed open`
- 限定：时间 `until 6 p.m.`；时间 `during that period`
- 上下文：its 指博物馆；that period 回指 3 月 3 日至 3 月 8 日。

### time_02

> The bridge reopened on the first Monday after June 10, 2025, and tolls resumed two days later.

参考事实：2 条；限定：2 项。

**事实 1　`bridge`**

- 参考引用：`The bridge reopened on the first Monday after June 10, 2025`
- 全部位置：[0, 59)（1 处）
- 核心证据：`The bridge reopened`
- 限定：时间 `on the first Monday after June 10, 2025`

**事实 2　`tolls`**

- 参考引用：`tolls resumed two days later`
- 全部位置：[65, 93)（1 处）
- 核心证据：`tolls resumed`
- 限定：时间 `two days later`
- 上下文：two days later 以桥梁重开时间为参照；不擅自换算绝对日期。

### time_03

> By the end of 2024, River Labs had hired 20 staff, but its hiring freeze lasted from January through March 2025.

参考事实：2 条；限定：3 项。

**事实 1　`staff`**

- 参考引用：`By the end of 2024, River Labs had hired 20 staff`
- 全部位置：[0, 49)（1 处）
- 核心证据：`River Labs had hired`、`staff`
- 限定：时间 `By the end of 2024`；数量 `20`

**事实 2　`freeze`**

- 参考引用：`its hiring freeze lasted from January through March 2025`
- 全部位置：[55, 111)（1 处）
- 核心证据：`its hiring freeze lasted`
- 限定：时间 `from January through March 2025`
- 上下文：its 指 River Labs。

### time_04

> The clinic will be closed next Friday, and appointments booked before October 1 remain valid until October 31.

参考事实：2 条；限定：4 项。

**事实 1　`future_closure`**

- 参考引用：`The clinic will be closed next Friday`
- 全部位置：[0, 37)（1 处）
- 核心证据：`The clinic`、`be closed`
- 限定：模态 `will`；时间 `next Friday`

**事实 2　`appointments`**

- 参考引用：`appointments booked before October 1 remain valid until October 31`
- 全部位置：[43, 109)（1 处）
- 核心证据：`appointments`、`remain valid`
- 限定：限制条件 `booked before October 1`；时间 `until October 31`
