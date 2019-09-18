"""
Copyright (c) 2019 Tobias Winchen <twinchen@mpifr-bonn.mpg.de>

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
from mpikat.utils.process_tools import ManagedProcess, command_watcher
from mpikat.utils.process_monitor import SubprocessMonitor
from mpikat.utils.sensor_watchdog import SensorWatchdog
from mpikat.utils.db_monitor import DbMonitor
from mpikat.utils.mkrecv_stdout_parser import MkrecvSensors
from mpikat.effelsberg.edd.pipeline.pipeline_register import register_pipeline
from mpikat.effelsberg.edd.pipeline.EDDPipeline import EDDPipeline, launchPipelineServer
import mpikat.utils.numa as numa

from katcp import Sensor, AsyncReply, FailReply
#from katcp.kattypes import request, return_reply, Int, Str

from tornado.gen import coroutine

import os
import time
import logging
import coloredlogs
import json
import tempfile
import threading

log = logging.getLogger("mpikat.effelsberg.edd.pipeline.CriticalPFBPipeline")
log.setLevel('DEBUG')

POLARIZATIONS = ["polarization_0", "polarization_1"]

DEFAULT_CONFIG = {
        "input_bit_depth" : 12,                             # Input bit-depth
        "samples_per_heap": 4096,                           # this needs to be consistent with the mkrecv configuration
        "samples_per_block": 64 * 1024 * 1024,              # sampels per buffer block
        "enabled_polarizations" : ["polarization_1"],
        "sample_clock" : 2600000000,
        "sync_time" : 1562662573.0,
        "fft_length": 128,
        "ntaps": 4,
        "output_bit_depth" : 8,                             # Output bit-depth (2,4,8,16,32)
        "null_output": False,                               # Disable sending of data for testing purposes
        "dummy_input": False,                               # Use dummy input instead of mkrecv process.
        "log_level": "debug",

        "output_rate_factor": 1.10,                         # True output date rate is multiplied by this factor for sending.

        "polarization_0" :
        {
            "ibv_if": "10.10.1.10",
            "mcast_sources": "225.0.0.152+3",
            "mcast_dest": "225.0.0.182",
            "port_rx": "7148",
            "port_tx": "7152",
            "dada_key": "dada",                             # output keys are the reverse!
            "numa_node": "1",                               # we only have one ethernet interface on numa node 1
        },
         "polarization_1" :
        {
            "ibv_if": "10.10.1.11",
            "mcast_sources": "225.0.0.156+3",
            "mcast_dest": "225.0.0.183",
            "port_rx": "7148",
            "port_tx": "7152",
            "dada_key": "dadc",
            "numa_node": "1",                               # we only have one ethernet interface on numa node 1
        }
    }

# static configuration for mkrec. all items that can be configured are passed
# via cmdline
mkrecv_header = """
## Dada header configuration
HEADER          DADA
HDR_VERSION     1.0
HDR_SIZE        4096
DADA_VERSION    1.0

## MKRECV configuration
PACKET_SIZE         8400
IBV_VECTOR          -1          # IBV forced into polling mode
IBV_MAX_POLL        10
BUFFER_SIZE         128000000

DADA_MODE           4    # The mode, 4=full dada functionality

SAMPLE_CLOCK_START  0 # This is updated with the sync-time of the packetiser to allow for UTC conversion from the sample clock

NTHREADS            32
NHEAPS              64
NGROUPS_TEMP        65536

#SPEAD specifcation for EDD packetiser data stream
NINDICES            1      # Although there is more than one index, we are only receiving one polarisation so only need to specify the time index

# The first index item is the running timestamp
IDX1_ITEM           0      # First item of a SPEAD heap

"""

# static configuration for mksend. all items that can be configured are passed
# via cmdline
mksend_header = """
HEADER          DADA
HDR_VERSION     1.0
HDR_SIZE        4096
DADA_VERSION    1.0

# MKSEND CONFIG
NETWORK_MODE  1
PACKET_SIZE 8400
IBV_VECTOR   -1          # IBV forced into polling mode
IBV_MAX_POLL 10

SYNC_TIME           unset  # Default value from mksend manual
SAMPLE_CLOCK        unset  # Default value from mksend manual
SAMPLE_CLOCK_START  0      # file Default value from mksend manual
UTC_START           unset  # Default value from mksend manual

#number of heaps with the same time stamp.
HEAP_COUNT 1
HEAP_ID_START   1
HEAP_ID_OFFSET  1
HEAP_ID_STEP    13

NITEMS          7
ITEM1_ID        5632    # timestamp, slowest index

ITEM2_ID        5633    # polarization

ITEM3_ID        5634    # fft_length
ITEM4_ID        5635    # n_taps

ITEM5_ID        5636    # sync_time

ITEM6_ID        5637    # sampling rate

ITEM7_ID        5638    # payload item (empty step, list, index and sci)
"""



@register_pipeline("CriticalPFBPipeline")
class CriticalPFBPipeline(EDDPipeline):
    """@brief critical PFB pipeline class."""
    VERSION_INFO = ("mpikat-edd-api", 0, 1)
    BUILD_INFO = ("mpikat-edd-implementation", 0, 1, "rc1")

    def __init__(self, ip, port, scpi_ip, scpi_port):
        """@brief initialize the pipeline."""
        EDDPipeline.__init__(self, ip, port, scpi_ip, scpi_port)

    def setup_sensors(self):
        """
        @brief Setup monitoring sensors
        """
        self._testSensor = Sensor.string(
            "test",
            description="test",
            initial_status=Sensor.NOMINAL)
        self.add_sensor(self._testSensor)

        self._control_mode_sensor = Sensor.string(
            "control-mode",
            description="The control mode for the EDD",
            default=self._control_mode,
            initial_status=Sensor.NOMINAL)
        self.add_sensor(self._control_mode_sensor)

        self._edd_config_sensor = Sensor.string(
            "current-config",
            description="The current configuration for the EDD backend",
            default=json.dumps(DEFAULT_CONFIG, indent=4),
            initial_status=Sensor.UNKNOWN)
        self.add_sensor(self._edd_config_sensor)
        self._edd_scpi_interface_addr_sensor = Sensor.string(
            "scpi-interface-addr",
            description="The SCPI interface address for this instance",
            default="{}:{}".format(self._scpi_ip, self._scpi_port),
            initial_status=Sensor.UNKNOWN)
        self.add_sensor(self._edd_scpi_interface_addr_sensor)
        self._device_status = Sensor.discrete(
            "device-status",
            description="Health status of device",
            params=self.DEVICE_STATUSES,
            default="ok",
            initial_status=Sensor.UNKNOWN)
        self.add_sensor(self._device_status)
        self._pipeline_sensor_status = Sensor.discrete(
            "pipeline-status",
            description="Status of the pipeline",
            params=self.PIPELINE_STATES,
            default="idle",
            initial_status=Sensor.UNKNOWN)
        self.add_sensor(self._pipeline_sensor_status)

        self._status_change_time = Sensor.string(
            "status-change-time",
            description="Time of last status change",
            default=time.ctime(),
            initial_status=Sensor.NOMINAL)
        self.add_sensor(self._status_change_time)

        self._output_rate_status = Sensor.float(
            "output-rate",
            description="Output data rate [Gbyte/s]",
            initial_status=Sensor.UNKNOWN)
        self.add_sensor(self._output_rate_status)

        self._polarization_sensors = {}
        for p in POLARIZATIONS:
            self._polarization_sensors[p] = {}
            self._polarization_sensors[p]["mkrecv_sensors"] = MkrecvSensors(p)
            for s in self._polarization_sensors[p]["mkrecv_sensors"].sensors.itervalues():
                self.add_sensor(s)
            self._polarization_sensors[p]["input-buffer-fill-level"] = Sensor.float(
                    "input-buffer-fill-level-{}".format(p),
                    description="Fill level of the input buffer for polarization{}".format(p),
                    params=[0, 1]
                    )
            self.add_sensor(self._polarization_sensors[p]["input-buffer-fill-level"])
            self._polarization_sensors[p]["input-buffer-total-write"] = Sensor.float(
                    "input-buffer-total-write-{}".format(p),
                    description="Total write into input buffer for polarization {}".format(p),
                    params=[0, 1]
                    )

            self.add_sensor(self._polarization_sensors[p]["input-buffer-total-write"])
            self._polarization_sensors[p]["output-buffer-fill-level"] = Sensor.float(
                    "output-buffer-fill-level-{}".format(p),
                    description="Fill level of the output buffer for polarization {}".format(p)
                    )
            self._polarization_sensors[p]["output-buffer-total-read"] = Sensor.float(
                    "output-buffer-total-read-{}".format(p),
                    description="Total read from output buffer for polarization {}".format(p)
                    )
            self.add_sensor(self._polarization_sensors[p]["output-buffer-total-read"])
            self.add_sensor(self._polarization_sensors[p]["output-buffer-fill-level"])


    @coroutine
    def _create_ring_buffer(self, bufferSize, blocks, key, numa_node):
         """
         Create a ring buffer of given size with given key on specified numa node.
         Adds and register an appropriate sensor to thw list
         """
         # always clear buffer first. Allow fail here
         yield command_watcher("dada_db -d -k {key}".format(key=key), allow_fail=True)

         cmd = "numactl --cpubind={numa_node} --membind={numa_node} dada_db -k {key} -n {blocks} -b {bufferSize} -p -l".format(key=key, blocks=blocks, bufferSize=bufferSize, numa_node=numa_node)
         log.debug("Running command: {0}".format(cmd))
         yield command_watcher(cmd)

         M = DbMonitor(key, self._buffer_status_handle)
         M.start()
         self._dada_buffers.append({'key': key, 'monitor': M})


    def _buffer_status_handle(self, status):
        """
        Process a change in the buffer status
        """
        for p in POLARIZATIONS:
            if status['key'] == self._config[p]['dada_key']:
                self._polarization_sensors[p]["input-buffer-total-write"].set_value(status['written'])
                self._polarization_sensors[p]["input-buffer-fill-level"].set_value(status['fraction-full'])
            elif status['key'] == self._config[p]['dada_key'][::-1]:
                self._polarization_sensors[p]["output-buffer-fill-level"].set_value(status['fraction-full'])
                self._polarization_sensors[p]["output-buffer-total-read"].set_value(status['read'])


    @coroutine
    def configure(self, config_json):
        """@brief destroy any ring buffer and create new ring buffer."""
        """
        @brief   Configure the EDD CCritical PFB

        @param   config_json    A JSON dictionary object containing configuration information

        @detail  The configuration dictionary is highly flexible. An example is below:
        """
        log.info("Configuring EDD backend for processing")
        log.debug("Configuration string: '{}'".format(config_json))
        if self.state != "idle":
            raise FailReply('Cannot configure pipeline. Pipeline state {}.'.format(self.state))
        # alternatively we should automatically deconfigure
        #yield self.deconfigure()

        self.state = "configuring"
        # Merge retrieved config into default via recursive dict merge
        def __updateConfig(oldo, new):
            old = oldo.copy()
            for k in new:
                if isinstance(old[k], dict):
                    old[k] = __updateConfig(old[k], new[k])
                else:
                    old[k] = new[k]
            return old

        if isinstance(config_json, str):
            cfg = json.loads(config_json)
        elif isinstance(config_json, dict):
            cfg = config_json
        else:
            self.state = "idle"     # no states changed
            raise FailReply("Cannot handle config type {}. Config has to bei either json formatted string or dict!".format(type(config_json)))
        try:
            self._config = __updateConfig(DEFAULT_CONFIG, cfg)
        except KeyError as error:
            self.state = "idle"     # no states changed
            raise FailReply("Unknown configuration option: {}".format(str(error)))


        cfs = json.dumps(self._config, indent=4)
        log.info("Received configuration:\n" + cfs)
        self._edd_config_sensor.set_value(cfs)

        # calculate input buffer parameters
        self.input_heapSize =  self._config["samples_per_heap"] * self._config['input_bit_depth'] / 8
        nHeaps = self._config["samples_per_block"] / self._config["samples_per_heap"]
        input_bufferSize = nHeaps * (self.input_heapSize)
        log.info('Input dada parameters created from configuration:\n\
                heap size:        {} byte\n\
                heaps per block:  {}\n\
                buffer size:      {} byte'.format(self.input_heapSize, nHeaps, input_bufferSize))

        # calculate output buffer parameters
        nSlices = max(self._config["samples_per_block"] / self._config['fft_length'], 1)
        nChannels = self._config['fft_length'] / 2
        # on / off spectrum  + one side channel item per spectrum
        output_bufferSize = nSlices * 2 * nChannels * self._config['output_bit_depth'] / 8
        output_heapSize = output_bufferSize / 8
        #output_bufferSize

        rate = output_bufferSize * float(self._config['sample_clock']) / self._config["samples_per_block"] # in spead documentation BYTE per second and not bit!
        rate *= self._config["output_rate_factor"]        # set rate to (100+X)% of expected rate
        self._output_rate_status.set_value(rate / 1E9)


        log.info('Output parameters calculated from configuration:\n\
                spectra per block:  {} \n\
                nChannels:          {} \n\
                buffer size:        {} byte \n\
                heap size:          {} byte\n\
                rate ({:.0f}%):        {} Gbps'.format(nSlices, nChannels, output_bufferSize, output_heapSize, self._config["output_rate_factor"] * 100, rate / 1E9))
        self._subprocessMonitor = SubprocessMonitor()

        for i, k in enumerate(self._config['enabled_polarizations']):
            numa_node = self._config[k]['numa_node']

            # configure dada buffer
            bufferName = self._config[k]['dada_key']
            yield self._create_ring_buffer(input_bufferSize, 64, bufferName, numa_node)

            ofname = bufferName[::-1]
            # we write nSlice blocks on each go
            yield self._create_ring_buffer(output_bufferSize, 64, ofname, numa_node)

            # Configure + launch
            # here should be a smarter system to parse the options from the
            # controller to the program without redundant typing of options
            physcpu = numa.getInfo()[numa_node]['cores'][0]
            cmd = "taskset {physcpu} pfb --input_key={dada_key} --inputbitdepth={input_bit_depth} --fft_length={fft_length} --ntaps={ntaps}   -o {ofname} --log_level={log_level} --outputbitdepth={output_bit_depth} --output_type=dada".format(dada_key=bufferName, ofname=ofname, heapSize=self.input_heapSize, numa_node=numa_node, physcpu=physcpu, **self._config)
            log.debug("Command to run: {}".format(cmd))

            cudaDevice = numa.getInfo()[self._config[k]["numa_node"]]["gpus"][0]
            cli = ManagedProcess(cmd, env={"CUDA_VISIBLE_DEVICES": cudaDevice})
            self._subprocessMonitor.add(cli, self._subprocess_error)
            self._subprocesses.append(cli)

            if not self._config["null_output"]:
                mksend_header_file = tempfile.NamedTemporaryFile(delete=False)
                mksend_header_file.write(mksend_header)
                mksend_header_file.close()

                cfg = self._config.copy()
                cfg.update(self._config[k])


                timestep = input_bufferSize * 8 / cfg['input_bit_depth']
                physcpu = ",".join(numa.getInfo()[numa_node]['cores'][1:2])
                cmd = "taskset {physcpu} mksend --header {mksend_header} --dada-key {ofname} --ibv-if {ibv_if} --port {port_tx} --sync-epoch {sync_time} --sample-clock {sample_clock} --item1-step {timestep} --item2-list {polarization} --item3-list {fft_length} --item4-list {ntaps} --item6-list {sample_clock} --item5-list {sync_time} --rate {rate} --heap-size {heap_size} {mcast_dest}".format(mksend_header=mksend_header_file.name, timestep=timestep,
                        ofname=ofname, polarization=i, nChannels=nChannels, physcpu=physcpu,
                        rate=rate, heap_size=output_heapSize, **cfg)
                log.debug("Command to run: {}".format(cmd))

                mks = ManagedProcess(cmd)
                self._subprocessMonitor.add(mks, self._subprocess_error)
                self._subprocesses.append(mks)
            else:
                log.warning("Selected null output. Not sending data!")

        self._subprocessMonitor.start()
        self.state = "ready"


    @coroutine
    def capture_start(self, config_json=""):
        """@brief start the dspsr instance then turn on dada_junkdb instance."""
        log.info("Starting EDD backend")
        if self.state != "ready":
            raise FailReply("pipleine state is not in state = ready, but in state = {} - cannot start the pipeline".format(self.state))
            #return

        self.state = "starting"
        try:
            mkrecvheader_file = tempfile.NamedTemporaryFile(delete=False)
            log.debug("Creating mkrec header file: {}".format(mkrecvheader_file.name))
            mkrecvheader_file.write(mkrecv_header)
            # DADA may need this
            mkrecvheader_file.write("NBIT {}\n".format(self._config["input_bit_depth"]))
            mkrecvheader_file.write("HEAP_SIZE {}\n".format(self.input_heapSize))

            mkrecvheader_file.write("\n#OTHER PARAMETERS\n")
            mkrecvheader_file.write("samples_per_block {}\n".format(self._config["samples_per_block"]))

            mkrecvheader_file.write("\n#PARAMETERS ADDED AUTOMATICALLY BY MKRECV\n")
            mkrecvheader_file.close()

            for i, k in enumerate(self._config['enabled_polarizations']):
                cfg = self._config.copy()
                cfg.update(self._config[k])
                if not self._config['dummy_input']:
                    numa_node = self._config[k]['numa_node']
                    physcpu = ",".join(numa.getInfo()[numa_node]['cores'][2:7])
                    cmd = "taskset {physcpu} mkrecv_nt --quiet --header {mkrecv_header} --idx1-step {samples_per_heap} --dada-key {dada_key} \
                    --sync-epoch {sync_time} --sample-clock {sample_clock} \
                    --ibv-if {ibv_if} --port {port_rx} {mcast_sources}".format(mkrecv_header=mkrecvheader_file.name, physcpu=physcpu,
                            **cfg )
                    mk = ManagedProcess(cmd, stdout_handler=self._polarization_sensors[k]["mkrecv_sensors"].stdout_handler)
                else:
                    log.warning("Creating Dummy input instead of listening to network!")
                    cmd = "dada_junkdb -c 1 -R 1000 -t 3600 -k {dada_key} {mkrecv_header}".format(mkrecv_header=mkrecvheader_file.name,
                            **cfg )

                    mk = ManagedProcess(cmd)

                self.mkrec_cmd.append(mk)
                self._subprocessMonitor.add(mk, self._subprocess_error)

        except Exception as e:
            log.error("Error starting pipeline: {}".format(e))
            self.state = "error"
        else:
            self.state = "running"
            self.__watchdogs = []
            for i, k in enumerate(self._config['enabled_polarizations']):
                wd = SensorWatchdog(self._polarization_sensors[k]["input-buffer-total-write"],
                        20,
                        self.watchdog_error)
                wd.start()
                self.__watchdogs.append(wd)


    @coroutine
    def capture_stop(self):
        """@brief stop the dada_junkdb and dspsr instances."""
        log.info("Stoping EDD backend")
        if self.state != 'running':
            log.warning("pipleine state is not in state = running but in state {}".format(self.state))
            # return
        log.debug("Stopping")
        for wd in self.__watchdogs:
            wd.stop_event.set()
        if self._subprocessMonitor is not None:
            self._subprocessMonitor.stop()

        # stop mkrec process
        log.debug("Stopping mkrecv processes ...")
        for proc in self.mkrec_cmd:
            proc.terminate()
        # This will terminate also the edd gpu process automatically

        yield self.deconfigure()


    @coroutine
    def deconfigure(self):
        """@brief deconfigure the dspsr pipeline."""
        log.info("Deconfiguring EDD backend")
        if self.state == 'runnning':
            yield self.capture_stop()

        self.state = "deconfiguring"
        if self._subprocessMonitor is not None:
            self._subprocessMonitor.stop()
        for proc in self._subprocesses:
            proc.terminate()

        self.mkrec_cmd = []

        log.debug("Destroying dada buffers")
        for k in self._dada_buffers:
            k['monitor'].stop()
            cmd = "dada_db -d -k {0}".format(k['key'])
            log.debug("Running command: {0}".format(cmd))
            yield command_watcher(cmd)

        self._dada_buffers = []
        self.state = "idle"


if __name__ == "__main__":
    launchPipelineServer(CriticalPFBPipeline)
