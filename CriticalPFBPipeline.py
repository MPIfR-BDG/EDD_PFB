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
from mpikat.effelsberg.edd.edd_scpi_interface import EddScpiInterface
import mpikat.utils.numa as numa

from katcp import Sensor, AsyncDeviceServer, AsyncReply, FailReply
from katcp.kattypes import request, return_reply, Int, Str

import tornado
from tornado.gen import coroutine

import os
import time
import logging
import signal
from optparse import OptionParser
import coloredlogs
import json
import tempfile
import threading

log = logging.getLogger("mpikat.effelsberg.edd.pipeline.GatedSpectrometerPipeline")
log.setLevel('DEBUG')

PIPELINE_STATES = ["idle", "configuring", "ready",
                   "starting", "running", "stopping",
                   "deconfiguring", "error"]

POLARIZATIONS = ["polarization_0", "polarization_1"]

DEFAULT_CONFIG = {
        "input_bit_depth" : 12,                             # Input bit-depth
        "samples_per_heap": 4096,                           # this needs to be consistent with the mkrecv configuration
        "samples_per_block": 512 * 1024 * 1024,             # 512 Mega sampels per buffer block 
        "enabled_polarizations" : ["polarization_1"],
        "sample_clock" : 2600000000,
        "sync_time" : 1562662573.0,
        "fft_length": 128,
        "ntaps": 4,

        "null_output": False,                               # Disable sending of data for testing purposes
        "dummy_input": False,                               # Use dummy input instead of mkrecv process.
        "log_level": "debug",

        "output_rate_factor": 1.10,                         # True output date rate is multiplied by this factor for sending.

        "polarization_0" :
        {
            "ibv_if": "10.10.1.10",
            "mcast_sources": "225.0.0.152+3",
            "mcast_dest": "225.0.0.172 225.0.0.173",        #two destinations gate on/off
            "port_rx": "7148",
            "port_tx": "7152",
            "dada_key": "dada",                             # output keys are the reverse!
            "numa_node": "1",                               # we only have one ethernet interface on numa node 1
        },
         "polarization_1" :
        {
            "ibv_if": "10.10.1.11",
            "mcast_sources": "225.0.0.156+3",
            "mcast_dest": "225.0.0.184 225.0.0.185",        #two destinations, one for on, one for off
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

# Add side item to buffer
SCI_LIST            2
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
SAMPLE_CLOCK_START  0      # Default value from mksend manual
UTC_START           unset  # Default value from mksend manual

#number of heaps with the same time stamp.
HEAP_COUNT 1
HEAP_ID_START   1
HEAP_ID_OFFSET  1
HEAP_ID_STEP    13

NSCI            1
NITEMS          7
ITEM1_ID        5632    # timestamp, slowest index

ITEM2_ID        5633    # polarization

ITEM3_ID        5634    # fft_length
ITEM4_ID        5635    # n_taps

ITEM5_ID        5636    # sync_time

ITEM6_ID        5638    # sampling rate

ITEM7_ID        5640    # payload item (empty step, list, index and sci)
"""



@register_pipeline("CriticalPFBPipeline")
class CriticalPFBPipeline(AsyncDeviceServer):
    """@brief critical PFB pipeline class."""
    VERSION_INFO = ("mpikat-edd-api", 0, 1)
    BUILD_INFO = ("mpikat-edd-implementation", 0, 1, "rc1")
    DEVICE_STATUSES = ["ok", "degraded", "fail"]

    CONTROL_MODES = ["KATCP", "SCPI"]
    KATCP, SCPI = CONTROL_MODES

    def __init__(self, ip, port, scpi_ip, scpi_port):
        """@brief initialize the pipeline."""
        self.callbacks = set()
        self._state = "idle"
        self._sensors = []
        self._control_mode = self.KATCP
        self._scpi_ip = scpi_ip
        self._scpi_port = scpi_port
        self._scpi_interface = None
        self._config = None
        self._subprocesses = []
        self.mkrec_cmd = []
        self._dada_buffers = []
        self._subprocessMonitor = None
        super(CriticalPFBPipeline, self).__init__(ip, port) # Async device parent depends on setting e.g. _control_mode in child


    @property
    def sensors(self):
        return self._sensors


    def notify(self):
        """@brief callback function."""
        for callback in self.callbacks:
            callback(self._state, self)


    @property
    def state(self):
        """@brief property of the pipeline state."""
        return self._state


    @state.setter
    def state(self, value):
        self._state = value
        self._pipeline_sensor_status.set_value(self._state)
        self._status_change_time.set_value(time.ctime())
        self.notify()


    def start(self):
        """
        @brief    Start the server
        """
        super(CriticalPFBPipeline, self).start()
        self._scpi_interface = EddScpiInterface(
            self, self._scpi_ip, self._scpi_port, self.ioloop)


    def stop(self):
        """
        @brief    Stop the server
        """
        self._scpi_interface.stop()
        self._scpi_interface = None
        super(GatedSpectrometerPipeline, self).stop()


    def setup_sensors(self):
        """
        @brief Setup monitoring sensors
        """
        self._testSensor = Sensor.string(
            "test",
            description="test",
            initial_status=Sensor.NOMINAL)
        self.add_sensor(self._testSensor)

        x = numpy.zeros(16)
        self._testSensor.set_value(x.tostring())



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
            params=PIPELINE_STATES,
            default="idle",
            initial_status=Sensor.UNKNOWN)
        self.add_sensor(self._pipeline_sensor_status)

        self._status_change_time = Sensor.string(
            "status-change-time",
            description="Time of last status change",
            default=time.ctime(),
            initial_status=Sensor.NOMINAL)
        self.add_sensor(self._status_change_time)

        self._integration_time_status = Sensor.float(
            "integration-time",
            description="Integration time [s]",
            initial_status=Sensor.UNKNOWN)
        self.add_sensor(self._integration_time_status)

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


    @property
    def katcp_control_mode(self):
        return self._control_mode == self.KATCP


    @property
    def scpi_control_mode(self):
        return self._control_mode == self.SCPI


    @request(Str())
    @return_reply()
    def request_set_control_mode(self, req, mode):
        """
        @brief     Set the external control mode for the master controller

        @param     mode   The external control mode to be used by the server
                          (options: KATCP, SCPI)

        @detail    The EddMasterController supports two methods of external control:
                   KATCP and SCPI. The server will always respond to a subset of KATCP
                   commands, however when set to SCPI mode the following commands are
                   disabled to the KATCP interface:
                       - configure
                       - capture_start
                       - capture_stop
                       - deconfigure
                   In SCPI control mode the EddScpiInterface is activated and the server
                   will respond to SCPI requests.

        @return     katcp reply object [[[ !configure ok | (fail [error description]) ]]]
        """
        try:
            self.set_control_mode(mode)
        except Exception as error:
            return ("fail", str(error))
        else:
            return ("ok",)


    def set_control_mode(self, mode):
        """
        @brief     Set the external control mode for the master controller

        @param     mode   The external control mode to be used by the server
                          (options: KATCP, SCPI)
        """
        mode = mode.upper()
        if not mode in self.CONTROL_MODES:
            raise UnknownControlMode("Unknown mode '{}', valid modes are '{}' ".format(
                mode, ", ".join(self.CONTROL_MODES)))
        else:
            self._control_mode = mode
        if self._control_mode == self.SCPI:
            self._scpi_interface.start()
        else:
            self._scpi_interface.stop()
        self._control_mode_sensor.set_value(self._control_mode)

    def _decode_capture_stdout(self, stdout, callback):
        log.debug('{}'.format(str(stdout)))


    def _handle_execution_stderr(self, stderr, callback):
        log.info(stderr)


    def _subprocess_error(self, proc):
        """
        Sets the error state because proc has ended.
        """
        log.error("Errror handle called because subprocess {} ended with return code {}".format(proc.pid, proc.returncode))
        self._subprocessMonitor.stop()
        self.state =  "error"


    @request(Str())
    @return_reply()
    def request_configure(self, req, config_json):
        """
        @brief      Configure EDD to receive and process data

       @note       The configure command accepts the following options in json format:
                    - "input_bit_depth"       : The bit-depth of the input data [8 or 12].
                    - "samples_per_heap"      : Number of samples in every heap of the input data stream (default 4096)
                    - "samples_per_block"     : Size of the input buffer blockn in samples . This indirectly defines the maximum size of the output spectra.
                    - "enabled_polarizations" : Enabled polarizations [polarization_0, polarization_1].
                    - "sample_clock"          : Sampling frequency in Hertz.
                    - "sync_time"             : Syncronization time.
                    - "fft_length"            : Number of sampels used for every FFT.
                    - "naccumulate"           : Numebr of spectr to integrate.
                    - "output_bit_depth"      : Bit depth of the output spectra [8, 12 or 32]
                    - "input_level"           :
                    - "output_level"          :
                    - "null_output"           : Disabling sending of data for testing purposes [true, false]/
                    - "dummy_input"           : Use dummy input instead of data from mkrecv process [true, false].
                    - "log_level"             : Log level used for console output.
                    - "output_rate_factor"    : True output date rate is multiplied by this factor for sending.
                    - "polarization_0,1"      : Dict of options for every polarization:
                        - "ibv_if"            : Ip of the NIC.
                        - "mcast_sources"     : Multicast adresses used for input
                        - "mcast_dest"        : Two multicast adresses used for output of on/off spectra
                        - "port_rx"           : Port to use for receiving.
                        - "port_tx":          : Port used for transmission
                        - "dada_key"          : Hex Key of buffer used used for input. Teh reverse key is used for output, e.g. dada-adad.
                        - "numa_node"         : Numa node used for processing.


        @return     katcp reply object [[[ !configure ok | (fail [error description]) ]]]
        """
        if not self.katcp_control_mode:
            return ("fail", "Master controller is in control mode: {}".format(self._control_mode))

        @coroutine
        def configure_wrapper():
            try:
                yield self.configure(config_json)
            except FailReply as fr:
                log.error(str(fr))
                req.reply("fail", str(fr))
            except Exception as error:
                log.exception(str(error))
                req.reply("fail", str(error))
            else:
                req.reply("ok")
        self.ioloop.add_callback(configure_wrapper)
        raise AsyncReply


    @request()
    @return_reply()
    def request_reconfigure(self, req):
        """
        @brief      Configure the EDD using the last configuration.

        @return     katcp reply object [[[ !reconfigure ok | (fail [error description]) ]]]
        """
        if not self.katcp_control_mode:
            return ("fail", "Master controller is in control mode: {}".format(self._control_mode))

        @coroutine
        def reconfigure_wrapper():
            try:
                yield self.configure(self._config)
            except FailReply as fr:
                log.error(str(fr))
                req.reply("fail", str(fr))
            except Exception as error:
                log.exception(str(error))
                req.reply("fail", str(error))
            else:
                req.reply("ok")
        self.ioloop.add_callback(reconfigure_wrapper)
        raise AsyncReply


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
        @brief   Configure the EDD gated spectrometer

        @param   config_json    A JSON dictionary object containing configuration information

        @detail  The configuration dictionary is highly flexible. An example is below:
                 @code
                     {
                         "nbeams": 1,
                         "nchans": 2048,
                         "freq_res": "something in Hz"
                         "integration_time": 1.0,
                         "mc_address": "255.0.0.152+8"
                         "mc_port": 7148
                     }
                 @endcode
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
        input_bufferSize = nHeaps * (self.input_heapSize + 64 / 8)
        log.info('Input dada parameters created from configuration:\n\
                heap size:        {} byte\n\
                heaps per block:  {}\n\
                buffer size:      {} byte'.format(self.input_heapSize, nHeaps, input_bufferSize))

        # calculate output buffer parameters
        nSlices = max(self._config["samples_per_block"] / self._config['fft_length'] /  self._config['naccumulate'], 1)
        nChannels = self._config['fft_length'] / 2 + 1
        # on / off spectrum  + one side channel item per spectrum
        output_bufferSize = nSlices * (2 * nChannels * self._config['output_bit_depth'] / 8 + 2 * 8)

        output_heapSize = nChannels * self._config['output_bit_depth'] / 8
        integrationTime = self._config['fft_length'] * self._config['naccumulate']  / float(self._config["sample_clock"])
        self._integration_time_status.set_value(integrationTime)
        rate = output_heapSize / integrationTime # in spead documentation BYTE per second and not bit!
        rate *= self._config["output_rate_factor"]        # set rate to (100+X)% of expected rate
        self._output_rate_status.set_value(rate / 1E9)


        log.info('Output parameters calculated from configuration:\n\
                spectra per block:  {} \n\
                nChannels:          {} \n\
                buffer size:        {} byte \n\
                integrationTime :   {} s \n\
                heap size:          {} byte\n\
                rate ({:.0f}%):        {} Gbps'.format(nSlices, nChannels, output_bufferSize, integrationTime, output_heapSize, self._config["output_rate_factor"]*100, rate / 1E9))
        self._subprocessMonitor = SubprocessMonitor()

        for i, k in enumerate(self._config['enabled_polarizations']):
            numa_node = self._config[k]['numa_node']

            # configure dada buffer
            bufferName = self._config[k]['dada_key']
            yield self._create_ring_buffer(input_bufferSize, 64, bufferName, numa_node)

            ofname = bufferName[::-1]
            # we write nSlice blocks on each go
            yield self._create_ring_buffer(output_bufferSize, 8 * nSlices, ofname, numa_node)

            # Configure + launch gated spectrometer
            # here should be a smarter system to parse the options from the
            # controller to the program without redundant typing of options
            physcpu = numa.getInfo()[numa_node]['cores'][0]
            cmd = "taskset {physcpu} gated_spectrometer --nsidechannelitems=1 --input_key={dada_key} --speadheap_size={heapSize} --selected_sidechannel=0 --nbits={input_bit_depth} --fft_length={fft_length} --naccumulate={naccumulate} --input_level={input_level} --output_bit_depth={output_bit_depth} --output_level={output_level} -o {ofname} --log_level={log_level} --output_type=dada".format(dada_key=bufferName, ofname=ofname, heapSize=self.input_heapSize, numa_node=numa_node, physcpu=physcpu, **self._config)
            log.debug("Command to run: {}".format(cmd))

            cudaDevice = numa.getInfo()[self._config[k]["numa_node"]]["gpus"][0]
            gated_cli = ManagedProcess(cmd, env={"CUDA_VISIBLE_DEVICES": cudaDevice})
            self._subprocessMonitor.add(gated_cli, self._subprocess_error)
            self._subprocesses.append(gated_cli)

            if not self._config["null_output"]:
                mksend_header_file = tempfile.NamedTemporaryFile(delete=False)
                mksend_header_file.write(mksend_header)
                mksend_header_file.close()

                cfg = self._config.copy()
                cfg.update(self._config[k])

                nhops = len(self._config[k]['mcast_dest'].split())

                timestep = cfg["fft_length"] * cfg["naccumulate"]
                physcpu = ",".join(numa.getInfo()[numa_node]['cores'][1:2])
                cmd = "taskset {physcpu} mksend --header {mksend_header} --dada-key {ofname} --ibv-if {ibv_if} --port {port_tx} --sync-epoch {sync_time} --sample-clock {sample_clock} --item1-step {timestep} --item2-list {polarization} --item4-list {fft_length} --item6-list {sync_time} --item7-list {sample_clock} --item8-list {naccumulate} --rate {rate} --heap-size {heap_size} --nhops {nhops} {mcast_dest}".format(mksend_header=mksend_header_file.name, timestep=timestep,
                        ofname=ofname, polarization=i, nChannels=nChannels, physcpu=physcpu, integrationTime=integrationTime,
                        rate=rate, nhops=nhops, heap_size=output_heapSize, **cfg)
                log.debug("Command to run: {}".format(cmd))

                mks = ManagedProcess(cmd)
                self._subprocessMonitor.add(mks, self._subprocess_error)
                self._subprocesses.append(mks)
            else:
                log.warning("Selected null output. Not sending data!")

        self._subprocessMonitor.start()
        self.state = "ready"


    @request()
    @return_reply()
    def request_capture_start(self, req):
        """
        @brief      Start the EDD backend processing

        @note       This method may be updated in future to pass a 'scan configuration' containing
                    source and position information necessary for the population of output file
                    headers.

        @note       This is the KATCP wrapper for the capture_start command

        @return     katcp reply object [[[ !capture_start ok | (fail [error description]) ]]]
        """
        if not self.katcp_control_mode:
            return ("fail", "Master controller is in control mode: {}".format(self._control_mode))

        @coroutine
        def start_wrapper():
            try:
                yield self.capture_start()
            except FailReply as fr:
                log.error(str(fr))
                req.reply("fail", str(fr))
            except Exception as error:
                log.exception(str(error))
                req.reply("fail", str(error))
            else:
                req.reply("ok")
        self.ioloop.add_callback(start_wrapper)
        raise AsyncReply


    @coroutine
    def request_halt(self, req, msg):
        """
        Halts the process. Reimplemnetation of base class halt without timeout as this crash
        """
        if self.state == "running":
            yield self.capture_stop()
        yield self.deconfigure()
        self.ioloop.stop()
        req.reply("Server has stopepd - ByeBye!")
        raise AsyncReply


    def watchdog_error(self):
        """
        @brief Set error mode requested by watchdog.
        """
        log.error("Error state requested by watchdog!")
        self.state = "error"


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
                        10 * self._integration_time_status.value(),
                        self.watchdog_error)
                wd.start()
                self.__watchdogs.append(wd)


    @request()
    @return_reply()
    def request_capture_stop(self, req):
        """
        @brief      Stop the EDD backend processing

        @note       This is the KATCP wrapper for the capture_stop command

        @return     katcp reply object [[[ !capture_stop ok | (fail [error description]) ]]]
        """
        if not self.katcp_control_mode:
            return ("fail", "Master controller is in control mode: {}".format(self._control_mode))

        @coroutine
        def stop_wrapper():
            try:
                yield self.capture_stop()
            except Exception as error:
                log.exception(str(error))
                req.reply("fail", str(error))
            else:
                req.reply("ok")
        self.ioloop.add_callback(stop_wrapper)
        raise AsyncReply


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
        # This will terminate also the gated spectromenter automatically

        yield self.deconfigure()


    @request()
    @return_reply()
    def request_deconfigure(self, req):
        """
        @brief      Deconfigure the EDD backend.

        @note       This is the KATCP wrapper for the deconfigure command

        @return     katcp reply object [[[ !deconfigure ok | (fail [error description]) ]]]
        """
        if not self.katcp_control_mode:
            return ("fail", "Master controller is in control mode: {}".format(self._control_mode))

        @coroutine
        def deconfigure_wrapper():
            try:
                yield self.deconfigure()
            except Exception as error:
                log.exception(str(error))
                req.reply("fail", str(error))
            else:
                req.reply("ok")
        self.ioloop.add_callback(deconfigure_wrapper)
        raise AsyncReply


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


@coroutine
def on_shutdown(ioloop, server):
    log.info("Shutting down server")
    yield server.stop()
    ioloop.stop()


if __name__ == "__main__":
    usage = "usage: %prog [options]"
    parser = OptionParser(usage=usage)
    parser.add_option('-H', '--host', dest='host', type=str, default='localhost',
                      help='Host interface to bind to')
    parser.add_option('-p', '--port', dest='port', type=int, default=1235,
                      help='Port number to bind to')
    parser.add_option('', '--scpi-interface', dest='scpi_interface', type=str,
                      help='The interface to listen on for SCPI requests',
                      default="")
    parser.add_option('', '--scpi-port', dest='scpi_port', type=int,
                      help='The port number to listen on for SCPI requests')
    parser.add_option('', '--scpi-mode', dest='scpi_mode', action="store_true",
                      help='Activate the SCPI interface on startup')
    parser.add_option('', '--log-level', dest='log_level', type=str,
                      help='Port number of status server instance', default="INFO")
    (opts, args) = parser.parse_args()
    logging.getLogger().addHandler(logging.NullHandler())
    logger = logging.getLogger('mpikat')
    logger.setLevel(opts.log_level.upper())

    log.setLevel(opts.log_level.upper())
    coloredlogs.install(
        fmt=("[ %(levelname)s - %(asctime)s - %(name)s "
             "- %(filename)s:%(lineno)s] %(message)s"),
        level=opts.log_level.upper(),
        logger=logger)
    ioloop = tornado.ioloop.IOLoop.current()
    log.info("Starting GatedSpectrometerPipeline instance")
    server = GatedSpectrometerPipeline(
        opts.host, opts.port,
        opts.scpi_interface, opts.scpi_port)
    log.info("Created GatedSpectrometerPipeline instance")
    signal.signal(
        signal.SIGINT, lambda sig, frame: ioloop.add_callback_from_signal(
            on_shutdown, ioloop, server))

    def start_and_display():
        log.info("Starting GatedSpectrometerPipeline server")
        server.start()
        log.debug("Started GatedSpectrometerPipeline server")
        if opts.scpi_mode:
            log.debug("SCPI mode")
            server.set_control_mode(server.SCPI)
        log.info(
            "Listening at {0}, Ctrl-C to terminate server".format(
                server.bind_address))\

    ioloop.add_callback(start_and_display)
    ioloop.start()
