#! /usr/bin/ruby
require 'base64'
require 'open-uri'

OUTPUT_DIR = "output"

img_counter = 0

STDIN.each_line { |line|
  img_counter += 1

  if line.start_with? 'data'
    img_format, encoding, data = line.match('data:image/(\w+);(\w+),(.+)').captures
    if img_format and encoding and data
      if encoding == "base64"
        dec_data = Base64.decode64(data)
        File.open("#{OUTPUT_DIR}/#{img_counter}.#{img_format}", "wb") { |f| f.write(dec_data) }
      else
        puts "unknown encoding - #{encoding}"
      end
    else
      puts "illegal line #{line}"
    end
  else
    if line.match '^https://encrypted-tbn\d.gstatic.com/images'
      # assume jpeg(?)
      img_format = 'jpeg'
      File.open("#{OUTPUT_DIR}/#{img_counter}.#{img_format}", "wb") { |f| f.write(open(line).read) }
    else
      puts "unknown link - #{line}"
    end
  end
}

puts 'Done!'
