std::ofstream test;

  int length = 25*10; //fix
        float* list = (float*)malloc(length * sizeof(float));

        std::ifstream infile("arraysTTB");
        std::ofstream outfile;
        outfile.open("predictionsTTBZ.txt", std::ios_base::trunc);
        std::string line;
        std::string temp;

        outfile << "[" << std::endl;

        bool first = true;

        while (std::getline(infile, line))
        {
                int i = 0;
                std::stringstream s_stream(line);
                while (s_stream.good() && i < length) {
                        std::getline(s_stream, temp, ','); //get first string delimited by comma
                        list[i] = ::atof(temp.c_str());
                        i++;
                }

                float output[2];
                network(list, length, output);
                if(first){
                        first = false;
                }
                else{
                        outfile << ", " << std::endl;
                }
                outfile << "[" << output[0] << "]" ;
        }

        outfile << std::endl << "]" << std::endl;

        infile.close();
        outfile.close();

        free(list);

        std::cout << "ALL COMPLETE" << std::endl;
  return 0;
}