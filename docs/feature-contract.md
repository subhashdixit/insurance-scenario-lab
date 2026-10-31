# Feature contract

All named columns are required in CSVs. A blank value means missing; extra columns are ignored. Defaults are synthetic training medians/modes. Numeric bounds constrain serving and mark unsupported source values missing during training. They do not define a population validity guarantee.

| Source field | Meaning / units | Accepted values |
|---|---|---|
| `age` | age | 18 to 100 |
| `sex` | sex | female, male |
| `bmi` | bmi | 10 to 70 |
| `children` | children | 0 to 10 (whole numbers) |
| `smoker` | smoker | no, yes |
| `region` | region | northeast, northwest, southeast, southwest |
