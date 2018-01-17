namespace :technician do

  desc 'Submit jobs so that they can be tested with the technician interface'

  task :schedule, [:operation_type_name] => [:environment] do |t, args|

    # define operation

    ot = OperationType.find_by_name(args[:operation_type_name])

    unless ot
      puts "Could not find #{args[:operation_type_name]}"
      exit 0
    end

    op = ot.operations.create status: "pending", user_id: 0, x: 0, y: 0, parent_id: 0

    # make plan
    plan = Plan.new(
      name: "Test Plan #{Date.today}",
      cost_limit: 100,
      status: "pending",
      user_id: 1
    )

    plan.budget_id = Budget.last.id
    plan.save
    op.associate_plan plan

    # submit plan
    planner = Planner.new plan.id
    planner.start

    # schedule job
    job, operations = ot.schedule([op], User.find(1), Group.find_by_name("technicians"))

    # launch brower window
    cmd = "/Applications/Google\\ Chrome.app/Contents/MacOS/Google\\ Chrome "
    cmd += "http://localhost:3000/krill/start?job=#{job.id}"
    system cmd

  end

end