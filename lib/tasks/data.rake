namespace :data do

  desc 'Upgrade data json fields to DataAssociations'

  task :upgrade => :environment do 
    Sample.includes(:items).all.each do |s| 
      puts "Sample #{s.id}: Upgrading #{s.items.length} items."
      s.items.each do |i|
        i.upgrade
      end
    end
  end

  task :downgrade => :environment do
    puts "Deleting #{DataAssociation.count} data associations"
    DataAssociation.destroy_all
  end

end