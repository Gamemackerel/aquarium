Installation
============

Prerequisites
--

* [Ruby on rails](http://rubyonrails.org/)
* [git](https://github.com/)
* A unix like environment, e.g. Mac OSX or Linux
* A MySQL server (optional for a full, production level installation)
    
Get the code
--

To use the latest stable release, go to the [release page](https://github.com/klavinslab/aquarium/releases) and get either the zip file or tar.gz file and unpackage it on your local computer.

If you want to use the bleeding edge code, you can clone the code to your local computer using git, as in 

	git clone https://github.com/klavinslab/aquarium

Install Protocols
--
	
Install at least one repository of protocols, such as aqualib, as follows:

	cd aquarium/repos
	git clone https://github.com/klavinslab/aqualib
	
You may also want to start your own repository of protocols. For example, do (from within my_protocols):

	mkdir my_protocols
	cd my_protocols

Then create a new file called hello.pl with the following code in it:

	step
		description: "Hello World"
	end
	
Finally, make the repo.

	git init
	git add .
	git commit -m "Initial commit"
	
These protocol libraries will be accessible via the "Protocols -> Under Version Control" menu.

Configure Aquarium
--

Go to aquarium/config/initializers and do

	cp aquarium_template.notrb aquarium.rb
	
Then edit aquarium.rb. You probably won't want to change anything initially.

Go to aquarium/config and do

	cp database_template.yml database.yml
	
Then edit database.yml to suit your local configuration. You probably don't need to change anything if you are running in "Development"" mode. Otherwise, you need to set up a MySQL server and associate a user name and password. 

Start Aquarium
--

Run

	rails s
	
to start aquarium. Then go do http://localhost:3000/ and see if it works!

This procedure should start a "Development" mode version with a local sqlite database in the db directory. This could be enough for some labs. However, the Klavins lab runs two versions of Aquarium using MySQL and [Phusion Passenger](https://www.phusionpassenger.com/index2). The first version is the "rehearsal" version, and the second version is the "production" version. This setup allows us to (a) periodically copy the databases from production to rehearsal servers via the ""Admin->Mirror Production" menu and (b) practice protocols without messing up our actual inventory. Details on installing Passenger can be found online.

Create an Account
--

Go to Admin->New User and make an account. This first account should be given administrative privilages so you can use it to make more accounts.

Run Hello World
--

Go to Protocols->Under Version Control and choose myprotocols/hello.pl and run the protocol.

Keep Going
--

Congratulations, you've installed Aquarium. Now go to Help and read about Plankton and Oyster.




	