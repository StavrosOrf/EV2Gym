\ Model ev_city
\ LP format - for model browsing. Use MPS format to capture full model detail.
Maximize
  abs_output_per_cs[0,0] + abs_output_per_cs[0,1] + [
   - 4 cs_mode_cs[0,0] * abs_output_per_cs[0,0]
   - 4 cs_mode_cs[0,1] * abs_output_per_cs[0,1] ] / 2 
Subject To
 cs_power_output[0,0]: - output[0,0,0] - output[1,0,0] + output_per_cs[0,0]
   = 0
 cs_power_output[0,1]: - output[0,0,1] - output[1,0,1] + output_per_cs[0,1]
   = 0
 bigM1.cs.0.0: - 1.833333333333333 cs_mode_cs[0,0] + output_per_cs[0,0]
   >= -1.833233333333333
 bigM2.cs.0.0: - 1.833333333333333 cs_mode_cs[0,0] + output_per_cs[0,0]
   <= 0
 bigM1.cs.0.1: - 1.833333333333333 cs_mode_cs[0,1] + output_per_cs[0,1]
   >= -1.833233333333333
 bigM2.cs.0.1: - 1.833333333333333 cs_mode_cs[0,1] + output_per_cs[0,1]
   <= 0
 bigM1.ev.0.0: output[0,0,0] - 0.0833333333333333 ev_mode[0,0,0]
   >= -0.0832333333333333
 bigM2.ev.0.0: output[0,0,0] - 0.0833333333333333 ev_mode[0,0,0] <= 0
 bigM1.ev.0.0: output[1,0,0] - 0.0833333333333333 ev_mode[1,0,0]
   >= -0.0832333333333333
 bigM2.ev.0.0: output[1,0,0] - 0.0833333333333333 ev_mode[1,0,0] <= 0
 bigM1.ev.0.1: output[0,0,1] - 0.0833333333333333 ev_mode[0,0,1]
   >= -0.0832333333333333
 bigM2.ev.0.1: output[0,0,1] - 0.0833333333333333 ev_mode[0,0,1] <= 0
 bigM1.ev.0.1: output[1,0,1] - 0.0833333333333333 ev_mode[1,0,1]
   >= -0.0832333333333333
 bigM2.ev.0.1: output[1,0,1] - 0.0833333333333333 ev_mode[1,0,1] <= 0
 cs_power_output1[0,0]: output_per_cs[0,0] >= 0
 cs_power_output1[0,1]: output_per_cs[0,1] >= 0
 cs_power_output2[0,0]: output_per_cs[0,0] <= 1.833333333333333
 cs_power_output2[0,1]: output_per_cs[0,1] <= 1.833333333333333
 ev_empty_port.0.0.0: output[0,0,0] = 0
 ev_empty_port_energy.0.0.0: energy[0,0,0] = 0
 ev_empty_port.1.0.0: output[1,0,0] = 0
 ev_empty_port_energy.1.0.0: energy[1,0,0] = 0
 ev_empty_port.0.0.1: output[0,0,1] = 0
 ev_empty_port_energy.0.0.1: energy[0,0,1] = 0
 ev_empty_port.1.0.1: output[1,0,1] = 0
 ev_empty_port_energy.1.0.1: energy[1,0,1] = 0
 ev_energy_level1[0,0,0]: energy[0,0,0] >= 0
 ev_energy_level1[0,0,1]: energy[0,0,1] >= 0
 ev_energy_level1[1,0,0]: energy[1,0,0] >= 0
 ev_energy_level1[1,0,1]: energy[1,0,1] >= 0
 ev_energy_level2[0,0,0]: energy[0,0,0] <= 1
 ev_energy_level2[0,0,1]: energy[0,0,1] <= 1
 ev_energy_level2[1,0,0]: energy[1,0,0] <= 1
 ev_energy_level2[1,0,1]: energy[1,0,1] <= 1
 ev_departure_energy.0.0.0: energy[0,0,0] >= 1
 ev_departure_energy.1.0.0: energy[1,0,0] >= 1
 ev_departure_energy.0.0.1: energy[0,0,1] >= 1
 ev_departure_energy.1.0.1: energy[1,0,1] >= 1
 ev_ch_power1[0,0,0]: [ ] >= 0
 ev_ch_power1[0,0,1]: [ ] >= 0
 ev_ch_power1[1,0,0]: [ ] >= 0
 ev_ch_power1[1,0,1]: [ ] >= 0
 ev_ch_power2[0,0,0]: [ ] <= 0
 ev_ch_power2[0,0,1]: [ ] <= 0
 ev_ch_power2[1,0,0]: [ ] <= 0
 ev_ch_power2[1,0,1]: [ ] <= 0
 ev_dis_power1[0,0,0]: [ ] >= 0
 ev_dis_power1[0,0,1]: [ ] >= 0
 ev_dis_power1[1,0,0]: [ ] >= 0
 ev_dis_power1[1,0,1]: [ ] >= 0
 ev_dis_power2[0,0,0]: [ ] <= 0
 ev_dis_power2[0,0,1]: [ ] <= 0
 ev_dis_power2[1,0,0]: [ ] <= 0
 ev_dis_power2[1,0,1]: [ ] <= 0
 cs_mode1.0.0: cs_mode_cs[0,0] = 1 -> output_per_cs[0,0] >= 0
 cs_mode0.0.0: cs_mode_cs[0,0] = 0 -> output_per_cs[0,0] <= 0
 cs_mode1.0.1: cs_mode_cs[0,1] = 1 -> output_per_cs[0,1] >= 0
 cs_mode0.0.1: cs_mode_cs[0,1] = 0 -> output_per_cs[0,1] <= 0
 ev_mode1.0.0: ev_mode[0,0,0] = 1 -> output[0,0,0] >= 0
 ev_mode0.0.0: ev_mode[0,0,0] = 0 -> output[0,0,0] <= 0
 ev_mode1.0.0: ev_mode[1,0,0] = 1 -> output[1,0,0] >= 0
 ev_mode0.0.0: ev_mode[1,0,0] = 0 -> output[1,0,0] <= 0
 ev_mode1.0.1: ev_mode[0,0,1] = 1 -> output[0,0,1] >= 0
 ev_mode0.0.1: ev_mode[0,0,1] = 0 -> output[0,0,1] <= 0
 ev_mode1.0.1: ev_mode[1,0,1] = 1 -> output[1,0,1] >= 0
 ev_mode0.0.1: ev_mode[1,0,1] = 0 -> output[1,0,1] <= 0
Bounds
Binaries
 ev_mode[0,0,0] ev_mode[0,0,1] ev_mode[1,0,0] ev_mode[1,0,1]
 cs_mode_cs[0,0] cs_mode_cs[0,1]
General Constraints
 abs_power_output_per_cs[0,0]: abs_output_per_cs[0,0] = ABS (
   output_per_cs[0,0] )
 abs_power_output_per_cs[0,1]: abs_output_per_cs[0,1] = ABS (
   output_per_cs[0,1] )
End
