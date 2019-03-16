# SIH-India-Vs-Pollution
Project for SmartIndiaHackathon2019
Tesla2.0

Backend Work is done in PHP by other team-members in separate repository.
https://github.com/Rohit599/SIH19

Progressive Web-App Portal for India Vs Pollution
==========================================

Cosmo Green: A platform for raising voice against Pollution

We will design a web platform for people where they can raise issues related to pollution nearby. They can submit blogs highlighting the situation, share pictures and data, mark location on the map. Other users can upvote, share or comment on these. There will be section for News from Health and Pollution ministries reflecting the news reports and tweets . There will be a Campaign section displaying the list of issues raised by communities, social workers and individuals. Issues will be displayed according to the no. of views and upvotes share & tweet on social media.On opening these issues, the user will be asked to sign a petition[given format] by entering his credentials. After a certain no. of petition signed, these petitions can be downloaded in pdf format and admin of this campaign can submit this petition to concerned ministry. There will also be an Impact Stories/Success Stories section highlighting the results of the campaign. This will motivate people to show active participation in the campaigns. Our portal also have separate section showing pollution levels in different geographic locations. Our website will also provide database of Annual and Daily Summary Report for Air and Water Pollution, Daily Air Quality Index(AQI) Report along with concentration plots. In addition to this, people can subscribe themselves for regular updates regarding any issue or pollution alert. Additionally, this platform will have full documentation about laws and rules & regulation for all pollution parameters.

Contents of Folder1
===================
* Main web app.

Server Requirements
====================
* PHP >= 5.6.4
* OpenSSL PHP Extension
* PDO PHP Extension
* Mbstring PHP Extension
* Tokenizer PHP Extension
* XML PHP Extension

Deployment
==========
* First install LAMP(Apache, PHP, MYSQL).
* Install Composer from https://getcomposer.org/download/
* Clone this repo or download it on your local system.
* Open composer and run this given command.
* composer install
* Rename the file .env.example to .env.
  cp .env.example .env
* Generate the Application key
  php artisan key:generate
* Set DB credentials, InfoConnect API URL and App Name in .env
* Migrate the database.
  php artisan migrate
* Seed the database
  php artisan db:seed
* Set project URL in app/Helpers/custom_url.php
  The default value has been already set as "http://localhost/Proj_mx02/public/"

Local Development Server
=========================
*To run this project on localhost
 php artisan serve
 This project will by default run on this server:
 http://localhost:8000/

*For more details
 php artisan serve --help

Contents of Folder2 (Machine Learning)
=====================================
* Text Spam Detector
* Image Classification
* Sentiment Analysis
* Keyword Extractor
* Chatbot

Deployment for Machine Learning
================================
* First install python3 from its official website.
* Then install pip3 in your system by command:
	[Ubuntu] sudo upgrade
		 sudo apt install python3-pip
* Install pytorch from its official website or enter command in ubuntu:
		pip3 install torch torchvision
* Run command: python3 app.py
* Run command[For Chatbot]: python3 robo.py

Hardware Requirements
=====================

Recommended: Intel Core i5 processor or greater, 4GB RAM, Intel HD Graphics Card, Windows 7 or later (32/64-bit), Mouse, Keyboard, Internet Connection.

Software Requirements
====================
* Operating system: Windows XP or later, Mac OS X 10.9.x or later, Linux, Android v5.0 or later,iOS.
* Web Browser: Internet Explorer 10-11/ Microsoft Edge, Google Chrome
* Adobe Flash Player
* Laravel Components
* MySql
* Python 3
* Pytorch

Functionalities and Innovation
================================
* Easy Login & Sign Up
* Heat Map categorising the type of pollution.
* Spam Detector easily retrained
* Adult Content Filtering
* Live Twitter Feeds
