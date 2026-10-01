# Aurora X2 Industrial Chiller — Service Guide (synthetic test fixture)

## 1. Overview

The Aurora X2 is a water-cooled industrial chiller designed for continuous operation in food processing plants, data centres and plastics moulding lines. It keeps process water within half a degree of its setpoint by modulating compressor speed rather than cycling the compressor on and off. Running the compressor continuously at partial load reduces mechanical wear and saves energy compared with older fixed-speed units, because start-up current surges are the main source of stress on the motor windings.

The unit is certified to ISO-9001 manufacturing standards and ships with controller firmware v2.3. Earlier firmware (v2.1 and below) does not support remote monitoring over Modbus TCP and should be upgraded before commissioning.

## 2. Keeping the unit cool in hot weather

When the ambient temperature in the plant room rises, the condenser has to reject heat into warmer air or water, and the chiller loses capacity. Operators often notice that the unit struggles on summer afternoons. The most effective remedies are improving ventilation around the cabinet, keeping at least one metre of clearance on the intake side, and cleaning the condenser fins every quarter so that airflow is not restricted by dust and lint. Shading the cabinet from direct sunlight also helps. None of these measures require a technician; they are routine housekeeping that any shift supervisor can carry out.

If the unit repeatedly trips on high head pressure despite good ventilation, the condenser water flow may be too low. Check that the strainer upstream of the condenser is clean and that isolation valves are fully open.

## 3. Fault codes

The controller displays a four-digit fault code prefixed with the letter E. Codes in the 1000 range are informational, the 3000 range are warnings, and the 4000 range stop the compressor.

- E-1002: Scheduled maintenance reminder. Acknowledge the message and book a service visit within 30 days.
- E-3107: Low refrigerant charge suspected. Superheat is above 12 K for more than ten minutes. A certified technician must leak-test the circuit before recharging with R-513A.
- E-4021: Compressor discharge temperature exceeded 115 °C. The compressor is locked out for 20 minutes. Repeated E-4021 faults within 24 hours usually indicate a failing expansion valve (part number EXV-88).
- E-4050: Loss of phase on the main supply. Check the three-phase supply and the main contactor K1 before resetting.

To reset a fault, hold the MENU and ENTER buttons together for five seconds. Faults in the 4000 range can only be reset after the underlying condition has cleared.

## 4. Water quality

Poor water quality is the most common cause of reduced heat transfer over the life of a chiller. Scale builds up on the evaporator tubes, acting as insulation, and the unit must work harder to reach the same setpoint. Operators should test the loop monthly. Hardness should stay below 150 ppm, and pH should stay between 7.0 and 8.5. A glycol concentration of 30 percent is recommended where the chiller serves outdoor pipework that may freeze in winter.

Biological growth such as algae can clog the strainers. A biocide dosing schedule agreed with your water treatment supplier will prevent this.

## 5. Warranty

The standard warranty covers parts for 24 months from commissioning and labour for 12 months. The warranty is void if the unit is operated with non-approved refrigerants or if the service log shows that quarterly condenser cleaning was skipped. Extended warranty plan AX-EXT-5 extends parts cover to five years and must be purchased within 90 days of commissioning.

## 6. Remote monitoring

With firmware v2.3 the controller exposes live readings over Modbus TCP on port 502. Register 40001 holds the leaving water temperature in tenths of a degree, register 40002 the entering water temperature, and register 40010 the active fault code. Integrators should poll no faster than once per second to avoid overloading the controller. The controller does not support encryption, so it must sit on an isolated operational network rather than the corporate LAN.

## 7. Decommissioning

At the end of its service life the refrigerant must be recovered by a certified technician and returned to an approved reclamation facility. Venting refrigerant to atmosphere is illegal in most jurisdictions. The steel cabinet and copper pipework are fully recyclable, and the controller board should be disposed of as electronic waste.
