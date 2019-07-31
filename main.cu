// Test programm for polyphase filterbank implementation
// 18 Jul 2019, Tobias Winchen

#include <iostream>
#include <fstream>
#include <iomanip>
#include "boost/program_options.hpp"
#include <ctime>

#include "CriticalPolyphaseFilterbank.h"

#include "psrdada_cpp/cli_utils.hpp"
#include "psrdada_cpp/multilog.hpp"
#include "psrdada_cpp/simple_file_writer.hpp"
#include "psrdada_cpp/dada_input_stream.hpp"
#include "psrdada_cpp/dada_output_stream.hpp"

const size_t ERROR_IN_COMMAND_LINE = 1;
const size_t SUCCESS = 0;
const size_t ERROR_UNHANDLED_EXCEPTION = 2;

int main(int argc, char** argv)
{
  key_t input_key;
  std::string output_type = "file";

  unsigned int nbits;
  unsigned int fft_length;
  unsigned int ntaps;
  std::string filtercoefficientsfile;

  char buffer[32];
  std::time_t now = std::time(NULL);
  std::tm *ptm = std::localtime(&now);
  std::strftime(buffer, 32, "%Y-%m-%d-%H:%M:%S.bp", ptm);
  std::string outputfilename(buffer);

  namespace po = boost::program_options;
  po::options_description desc("Options");

  desc.add_options()("help,h", "Print help messages");
  desc.add_options()(
        "input_key,i",
        po::value<std::string>()->default_value("dada")->notifier(
            [&input_key](std::string in) { input_key = psrdada_cpp::string_to_key(in); }),
        "The shared memory key for the dada buffer to connect to (hex "
        "string)");
  desc.add_options()(
        "output_type", po::value<std::string>(&output_type)->default_value(output_type),
        "output type [dada, file]. Default is file."
        );
  desc.add_options()(
        "output_key,o", po::value<std::string>(&outputfilename)->default_value(outputfilename),
        "The key of the output bnuffer / name of the output file to write spectra "
        "to");
  desc.add_options()("ntaps,t", po::value<unsigned int>(&ntaps)->required(),
                       "The numbr of taps");
  desc.add_options()("fft_length,n", po::value<unsigned int>(&fft_length)->required(),
                       "The length of the FFT to perform on the data");
  desc.add_options()("inbutbitdepth,b", po::value<unsigned int>(&nbits)->required(),
                       "The number of bits per sample in the "
                       "packetiser output (8 or 12)");
  desc.add_options()(
        "filtercoefficients,f", po::value<std::string>(&filtercoefficientsfile)->default_value(filtercoefficientsfile),
        "txt file with filter coefficents. If empty, Default Kaiser coefficients will be calculated.  "
        );
  desc.add_options()(
        "log_level", po::value<std::string>()->default_value("info")->notifier(
                         [](std::string level) { psrdada_cpp::set_log_level(level); }),
        "The logging level to use "
        "(debug, info, warning, "
        "error)");

  po::variables_map vm;
  try {
    po::store(po::parse_command_line(argc, argv, desc), vm);
    if (vm.count("help")) {
      std::cout << "CriticalPolyphaseFilterBank-- Read EDD data from a DADA buffer and applies an nchannel ntap PFB."
                << std::endl
                << desc << std::endl;
      return SUCCESS;
    }

    po::notify(vm);
    if (vm.count("output_type") && (!(output_type == "dada" || output_type == "file") ))
    {
      throw po::validation_error(po::validation_error::invalid_option_value, "output_type", output_type);
    }

  }
   catch (po::error &e) {
      std::cerr << "ERROR: " << e.what() << std::endl << std::endl;
      std::cerr << desc << std::endl;
      return ERROR_IN_COMMAND_LINE;
    }



  //float f = 1./3;
  //if (argc == 2)
  //  f = atof(argv[1]);

  size_t nSpectra = 16384 * 16;

	cudaStream_t stream;
  cudaStreamCreate( &stream );

  FilterCoefficientsType filterCoefficients(fft_length * ntaps);

  if (filtercoefficientsfile.empty())
  {
    // Window with a critical frequency at the number of channels. pialhpa = 8 is
    // a non-optimized choice.
    double pialhpa = 8.;
    double fc = 1./(fft_length / 2 + 1);
    BOOST_LOG_TRIVIAL(info) << "No filter coefficients provided. Calculating coefficients based on Kaiser Window with paramters:\n"
      << "    pi * alpha = " << pialhpa << std::endl 
      << "            fc = " << fc << " ( number of channels**(-1) )";
    calculateKaiserCoefficients(filterCoefficients, pialhpa, fc);
  }
  else
  {
    BOOST_LOG_TRIVIAL(info) << "Reading filtercoefficeints from file: " << filtercoefficientsfile;
    std::ifstream infile(filtercoefficientsfile.c_str());
    if (!infile.good())
      throw std::runtime_error( "EDD PFB: could not open file: " + filtercoefficientsfile);
    int i = 0;
    std::string line;
    while (std::getline(infile,line))
    {
      std::stringstream stream(line);
      if (stream.peek() == '#')
  			continue;
      double x;
      stream >> x;

      if (i < filterCoefficients.size())
      {
        filterCoefficients[i] = x;
        i++;
      }
      else
      {
        throw std::runtime_error( "EDD PFB: Too many filter coefficients in file: " + filtercoefficientsfile);
      }
    }
    if (i < filterCoefficients.size())
    {
      BOOST_LOG_TRIVIAL(error) << "EDD PFB: Not enough filter coefficients in file :" + filtercoefficientsfile << std::endl
        << "    -  Require " << filterCoefficients.size() << " values, got only " << i << " values!";
        throw std::runtime_error("EDD PFB: Not enough filter coefficients in file: " + filtercoefficientsfile);
    }
  }

  BOOST_LOG_TRIVIAL(info) << "Running with  output_type: " << output_type;
  psrdada_cpp::MultiLog log("PFB");
  if (output_type == "file")
  {
    psrdada_cpp::SimpleFileWriter sink(outputfilename);
    CriticalPolyphaseFilterbank<decltype(sink)> ppf(fft_length, ntaps, nSpectra, nbits, filterCoefficients, sink);
    psrdada_cpp::DadaInputStream<decltype(ppf)> istream(input_key, log, ppf);
    istream.start();
  }
  else if (output_type == "dada")
  {
    psrdada_cpp::DadaOutputStream sink(psrdada_cpp::string_to_key(outputfilename), log);
    CriticalPolyphaseFilterbank<decltype(sink)> ppf(fft_length, ntaps, nSpectra, nbits, filterCoefficients, sink);
    psrdada_cpp::DadaInputStream<decltype(ppf)> istream(input_key, log, ppf);
    istream.start();
  }
  else
  {
    throw std::runtime_error("Unknown oputput-type");
  }
  return SUCCESS;
}
