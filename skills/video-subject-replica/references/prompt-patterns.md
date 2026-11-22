# Conflict-Free Prompt Patterns for Seedance 2.5 Flash `edit` Mode

## Core Principles
1. **Explicit Asset Indexing**: Map `视频1` and `图片1..N` explicitly in the first paragraph. Studio refs define ONLY appearance/anatomy (ignore studio backdrop & scale); pixel-locked scene refs define ONLY scale/lighting integration (NEVER first frame, NEVER freeze frame).
2. **Release Conflicting Motion Priors**: Video edit models strongly inherit optical flow from the original subject. If the replacement has a different posture, height, limb count, or vehicle/creature geometry, explicitly write: *"允许彻底改变原主体的身高、姿态、肢体动作与外形轮廓，禁止残留原主体形态"*.
3. **Observable Mechanics Over Vague Adjectives**: Replace "自然运动" with concrete physical mechanics (e.g., alternating knee flexion, flowing silk sleeves, coiled serpentine dragon body undulating with floating cloud mist, grounded footfalls/wheels).
4. **Explicit Conflict Resolution Rule**: Always include: *"参考图与视频发生冲突时，背景环境、构图、镜头运动、色调与时间节奏严格服从视频1；替换主体的外观与结构严格服从图片1和图片2。"*

---

## Archetype 1: Character + Vehicle Replacement (e.g., Adult Cyclist -> Child on Tricycle)
- **Recommended Refs (4)**:
  1. `图片1`: Studio primary camera-angle view matching dominant shot angle (e.g., straight-behind low seated).
  2. `图片2`: Studio 3/4 structural view exposing wheels, cranks, and seated geometry.
  3. `图片3` & `图片4`: 2 pixel-locked scene keyframes at distinct moments with opposite leg/pedal phase.
- **Template**:
  > 编辑视频1：仅将视频1中的[原人物与原车辆]替换为图片1和图片2中的[新人物与新车辆]。图片1和图片2仅定义人物外观、坐姿与车辆机械结构，不采用其灰底背景或主体大小；图片3与图片4仅为局部替换后的尺度、光照与阴影融入参考，不作为首帧，不复制为静止画面。
  > 允许彻底改变原主体的身高、站姿腿部动作与原车辆轮廓，完全替换为[具体的紧凑坐姿/驾驶动作与车辆机械运动描述]，禁止残留原人物、头盔或原车辆结构。
  > 保持视频1每一时刻的行进路线、朝向与克制的画面占比（人物不放大）；完整保留视频1的镜头推进、转弯倾斜、前景视差与时间节奏。除替换区域、接地阴影与擦除露出的局部路面外，背景环境与色彩光影严格以视频1为准，不重绘背景、不改变曝光与白平衡。参考图与视频冲突时，背景、色调与运镜服从视频1，主体外观与结构服从图片1、图片2。全程一镜到底，无字幕，无水印。

---

## Archetype 2: Multi-Subject Replacement — Character + Mythical Creature / Monster
- **Recommended Refs (4–5)**:
  1. `图片1`: Clean studio reference of the replacement human character (e.g., Chinese woman — hair, face/profile, authentic clothing fabric, matching lighting direction).
  2. `图片2`: Clean studio reference of the Chinese mythical creature (e.g., 应龙/青龙、白泽、饕餮、穷奇、九尾狐、山海经神兽 — scale texture, horns/mane/antlers, silhouette, atmospheric mist/armor details).
  3. `图片3` & `图片4`: 2 pixel-locked scene keyframes showing BOTH replaced subjects together in the original shot at early + climax moments (locking relative scale between woman and creature, eye-line/interaction, and environmental lighting!).
- **Template**:
  > 编辑视频1：将视频1中的[原人物]替换为图片1中的[中国女性角色描述]，同时将视频1中的[原怪物]替换为图片2中的[中国传统神话神兽/瑞兽/妖兽描述]。图片1和图片2仅定义人物与神话生物的外观、服饰材质与生物结构，不采用其棚拍背景或比例；图片3和图片4仅为镜头内双主体尺度关系、位置与环境光影融入参考，不作为首帧，不复制为静止画面。
  > 允许彻底改变原怪物与原人物的外形轮廓、骨骼比例与局部动作细节，使[神话生物]呈现出真实有重量感的材质（鳞片/毛发/云纹/金石质感）与符合生物力学的连贯动态，使[中国女性]保持自然的东方五官、神态、衣袂物理摆动与连贯肢体动作；禁止残留原西式怪物轮廓或原角色外貌。
  > 严格保留视频1中双主体的空间相对位置、互动节奏、画面占比与镜头调度（跟拍/摇移/景别变化/剪辑节奏）；除替换主体、接触阴影与被遮挡恢复的局部背景外，场景建筑、地形、光线方向、空气透视与电影色调严格以视频1为准，不重新设计场景，不偏色，不磨平细节。参考图与视频冲突时，环境、构图、运镜与光影服从视频1，角色与神话生物设计服从图片1、图片2。无字幕，无水印。

---

## Archetype 3: Handheld Product / Commercial Object Replacement
- **Recommended Refs (2–3)**:
  1. `图片1`: Crisp studio packshot of target product (accurate logo/label/material).
  2. `图片2`: Optional angled studio view if the product rotates in hand.
  3. `图片3`: Pixel-locked scene keyframe showing exact finger grip, specular reflection, and scale in the original hand.
- **Key Addition**: Explicitly lock hands, fingers, nails, and grip pressure (*"完整保留视频1中手部肤色、手指姿态与握持遮挡关系，仅替换手中物体，保持瓶身/产品标签清晰可读不扭曲，高光反射与环境光一致"*).

---

## Archetype 4: Local Outfit / Prop Replacement (Same Person)
- **Recommended Refs (2)**:
  1. `图片1`: Clean garment/prop studio reference.
  2. `图片2`: Pixel-locked scene keyframe wearing/holding the item.
- **Key Addition**: Explicitly lock original identity, face, hair, body motion, and background (*"严格保持视频1人物的脸部特征、发型、表情与动作不变，仅将原服装/道具替换为图片1样式"*).
