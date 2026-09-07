on a revent one39 gas oven, the red reset button on the gas valve is it on 24 vdc or 120 vac. i have to reset the red reset button about once a day. look at the uploaded oven schematic. what can be the cause of the frequent reset of the gas valve 

---  

**Assessing the Inquiry**

I'm now zeroing in on the specifics. My focus is the user's inquiry, specifically related to the Revent ONE39 Gas Oven. I'm carefully analyzing the request for voltage details of the gas valve reset button. My initial assessment is complete; the critical piece of information that remains is this voltage rating. I will need to determine whether the button uses 24 VDC or 120 VAC.


**Analyzing the Schematics**

I've moved on to examining the schematics, specifically pages 1 and 2. Page 1 details the power inputs, motor circuits, and power supplies, including 24VDC and 12VDC options. Page 2, focusing on the 120-139VAC control and burner circuits, shows the power flow through fuse 5F. I am now looking at the "BURNER CONTROL 120VAC" section to correlate this information. The frequent reset is the key issue.


**Tracing the Power Flow**

I'm now tracing the power flow through the terminals. I've confirmed that X1 5 is indeed connected to 120VAC via the 4CR coil and safety loop, and I'm correlating this with the burner control section. I see the line from 5F goes to 7CR before entering the Burner Control at X1 5. I'm noting the function of each terminal, including the interlock and heat call. I'm also examining the FAULT BURNER coil (4CR) and the connection to the hot surface igniter and gas valve at terminal 10X 29, noting how this connects through a contact to X1. Finally, I see the high limit safety loop terminals.


**Pinpointing the Circuit**

I'm now identifying the control circuit associated with the burner, specifically focusing on the red reset button. I've located the `2TAS` temperature limit between `X3 4` and `10X 1 / X3 12`. I'm tracing the links and confirming its connection to `3CR`, while also looking at connections to `X3 17, 14, 11`. The control fan motor connections at `X5 1-5` are also helpful. Now, I will look to establish the power supply.

I can confirm the reset button operates on 120VAC, based on the burner control section. My next step will be to finalize the analysis of the electrical flow.


**Confirming the Power Source**

I've further investigated the 24VDC control section on page 3. The input fault burner relay contacts `4CR` feed into the oven interface board, along with `3CR`, and I am now focusing on the interface between these sections and the 120VAC burner control unit to confirm the reset button operates on line voltage. The reset circuitry is integrally connected into the line-voltage burner control module.


**Determining the Failure Modes**

I'm now diving into the potential causes for the frequent resets. I see the burner lockout stems from ignition failures or flame loss. I'm focusing on the Flame Rod, or the HSI, looking for carbon buildup or loose grounding that's causing issues.


**Investigating Grounding Issues**

I'm now zeroing in on a bad earth ground, specifically related to the burner, as this would cause intermittent flame sense loss, according to my initial thoughts. I've noted the grounding symbol. I'm also examining potential issues related to the hot surface igniter, or HSI. This investigation has moved to issues pertaining to the combustion air / blower, along with gas supply and valve issues.

**Analyzing Combusion Air**

I'm now investigating potential failures related to combustion. I'm taking a look at the fan motor and PWM/Tach feedback mechanisms, as shown on Page 2, examining the X5/X3 connectors. I am noting that if the tachometer (TACHO signal on X3 2 / X5 2) doesn't match the requested PWM speed (X3 6 / X5 4), it's a potential lockout, I am considering dust, lint, or flour buildup, as possible sources for TACH errors. This investigation has moved to issues pertaining to gas supply.


**Evaluating Interlocks**

I am now focusing on the interlocks to understand potential safety issues causing resets, and potential for improper operation. Specifically, I'm noting the main circulation fan interlock (1M), high-temperature limit (2TAS), and burner safety relay (7CR). I'm also considering flour dust buildup, as the air intake can soots the flame sensor.


Based on the provided wiring schematics for the **Revent ONE39 Gas Oven** (Drawing No. 391407), here are the answers regarding your gas valve reset button and troubleshooting steps:

---

### 1. Is the Red Reset Button 24 VDC or 120 VAC?

* **It operates on 120 VAC.**

#### Why (from the schematic):
* **Page 2 (Wiring Diagram Control 120–139 VAC):** Shows the **`BURNER CONTROL 120VAC`** module. The main line power to this control module, the Hot Surface Igniter (**HSI**), the gas valve output, and the internal lockout/reset circuitry are powered by **120 VAC** (fed through fuse **`5F`** from the 120V control supply).
* The **`BURNER CONTROL 24VDC`** section (terminals `X3`) is strictly used for low-voltage signal interfacing (combustion fan PWM speed modulation, tachometer feedback, safety limit monitoring, and signaling the **`4CR`** fault relay to the Oven Interface Board).

---

### 2. What Causes the Burner to Lock Out ~Once a Day?

A burner lockout (illuminating the red reset button) occurs when the burner ignition controller attempts to fire and either fails ignition during the safety time or loses flame detection during operation.

Because it trips only **once a day** (intermittent fault), check the following common culprits shown on the schematic:

---

#### A. Flame Sensor / Grounding Issue (*Most Common*)
* **Oxidation on the Flame Rod / HSI:** Over time, carbon or silica deposits build up on the flame rectification sensor or Hot Surface Igniter (**HSI** on Page 2). This weakens the micro-amp (µA) flame signal, causing the controller to intermittently think the flame went out.
* **Poor Earth Ground (`PE`):** Flame rectification relies on a continuous path back through the chassis ground (note the ground connection on Page 2 at terminal `X1 8` / burner body). A loose or corroded ground screw on the burner chassis will intermittently drop the flame signal.

---

#### B. Combustion Fan Motor & Tachometer Signal (Page 2, `X5` to `X3`)
* The combustion fan has a 5-wire interface with **PWM speed control** and **TACHO feedback** (`X3 2` / `X5 2`).
* **Flour Dust / Lint Build-up:** If bakery dust builds up on the combustion fan impeller, it creates drag. If the burner control detects that the fan RPM (TACHO) does not match the commanded PWM speed, it will initiate a safety lockout.
* *Check:* Clean the combustion air blower impeller and verify the wiring harness at `X5` / `X3`.

---

#### C. Gas Supply Pressure Drop (Peak Usage Times)
* If the oven locks out during a specific busy time of day (e.g., early morning bake when proofers, water heaters, and other ovens kick on simultaneously), the incoming dynamic gas pressure may momentarily dip below minimum operating pressure, causing flame failure on ignition.
* *Check:* Measure **dynamic gas pressure** at the gas valve manifold with all gas equipment firing simultaneously.

---

#### D. Hot Surface Igniter (HSI) Degradation
* Hot surface igniters increase in resistance as they age. An aging igniter may glow, but not reach sufficient temperature quickly enough to ignite the gas within the 4-to-7 second trial-for-ignition window.
* *Check:* Measure the cold resistance of the HSI element against manufacturer specs (replace if out of range).

---

#### E. Air Proving Switch (`1PS`) Intermittent Drop
* On Page 3 (terminals `J9/14` & `J7/12`) and Page 6, the **`1PS` Air Prover** monitors combustion airflow.
* A sticking microswitch, cracked silicone sensing tube, or lint in the pitot tube can cause the air switch contact to briefly flutter open, instantly cutting the burner cycle.

---

#### F. Intermittent Relay / Contactor Contacts (`1M` & `1CR`)
* On Page 2, line 3: The 120 VAC call-for-heat signal passes through **`1M` (Main Fan aux contact 13-14)** and **`1CR` (Heat Contactor contact 1-2)** before entering `X1 1`.
* If contact `1M (13-14)` or `1CR (1-2)` has pitted/worn contacts or is subject to oven vibration, a momentary interruption will cause an ignition retry or lockout.

---

### Recommended Action Plan:
1. **Clean the Flame Rod / Igniter:** Remove the burner assembly; clean the flame sensor rod with fine steel wool / Scotch-Brite (do not use sandpaper).
2. **Clean the Combustion Fan:** Blow out any flour dust from the blower wheel and motor housing.
3. **Verify Grounding:** Ensure the burner frame ground wire is clean and tight to bare metal.
4. **Inspect the Air Tubing:** Check the clear silicone tubing going to the air switch (`1PS`) for moisture or cracks.

---  

Does it makes sense the gas pressure at the gas valve outlet is negative just before ignition and goes to 0 pressure (i.e. recover from negative pressure) after ignition

---  
**Yes, this makes absolute sense and is completely normal** for this type of oven. 

The Revent ONE39 uses a **pre-mix combustion system with a zero-governor (1:1 air/gas ratio) gas valve and a Venturi mixer**, modulated by the variable-speed combustion blower shown on your schematic.

Here is exactly why you are seeing that pressure behavior and what it means for your daily lockout issue:

---

### Why You See Negative Pressure Before Ignition and ~0 Pressure After

1. **Before Ignition (Pre-Purge Phase):**
   * The combustion blower spins up to purge the chamber and prepare for ignition.
   * As the blower forces air through the **Venturi tube** (mixer), it creates a strong suction (vacuum/negative pressure) at the gas inlet port.
   * Because the gas valve solenoids are still closed, this suction pulls a vacuum directly on the outlet test port of the gas valve.
   * **Result:** Your manometer reads **negative pressure** (typically $-0.1$ to $-0.5\text{ in. w.c.}$ / $-25$ to $-120\text{ Pa}$).

2. **At Ignition and Running:**
   * The gas valve solenoids open. 
   * A "zero-governor" gas valve does not push gas into the burner with high positive pressure like a traditional atmospheric burner. Instead, it holds gas back until the suction from the blower draws it out.
   * Once the valve opens, gas flows to satisfy the suction, balancing the outlet pressure back to **near zero** (neutral pressure, typically between $-0.04\text{ in. w.c.}$ and $0.00\text{ in. w.c.}$, depending on factory offset tuning).
   * **Result:** The manometer recovers from negative and stabilizes at **$\sim0\text{ pressure}$**.

---

### How This Connects to Your Daily Lockout Issue

Because this system relies on precise air-to-gas vacuum balance rather than raw positive gas pressure, **even tiny variations can cause the ignition mixture to be too lean or too rich to light**, triggering the red reset button:

#### 1. Gas Valve "Offset" (Zero-Point) Out of Adjustment
* On zero-governor valves, there is an **offset adjustment screw** (often under a plastic cap or Torx cover).
* If the offset is drifting slightly lean, during pre-purge the suction might be a bit too strong, and when the valve cracks open, insufficient gas is pulled into the ignition zone to ignite in time.
* *Fix:* The burner's combustion ($O_2$ / $CO_2$ levels at low fire and high fire) must be verified and tuned using a **flue gas combustion analyzer**.

#### 2. Dirty Combustion Fan / Venturi
* On Page 2, the schematic shows the combustion blower speed is controlled via **PWM** and monitored via **TACHO** (`X5`/`X3`).
* If bakery flour dust settles on the fan blades or inside the Venturi throat, the air velocity at low ignition RPM changes. This changes the vacuum pulled on the gas valve during start-up, leading to an intermittent failed ignition.

#### 3. Air Reference Line Issues
* Many of these gas valves have a small clear silicone reference tube connecting the air mixer/blower to the gas valve regulator diaphragm.
* If that tube is cracked, has moisture inside, or is slightly loose, the valve will not track the blower suction correctly during start-up.

---

### Summary
Your pressure reading confirms your oven has a **pneumatic pre-mix burner system**. The negative-to-zero pressure swing is correct, but the daily lockout strongly points toward:
* **The gas valve low-fire offset needing a slight adjustment** (via a combustion analyzer).
* **Flour dust inside the combustion blower / Venturi mixer.**
* **An aging Hot Surface Igniter (HSI) / dirty flame sensor** struggling to light a finely tuned air-fuel mix.